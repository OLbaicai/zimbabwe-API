import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageOps

API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME   = "HunyuanOCR"
API_KEY      = "sk-placeholder-key"

IMAGE_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID\金属ID图像")
OCR_DIR    = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID\WAY2\金属ID-MAIN-OUT")
DEAL_DIR   = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID\WAY2\金属ID-MAIN-1-2-DEAL-OUT")

MAX_COUNT   = 0
TIMEOUT     = 120
MAX_RETRIES = 2
RETRY_DELAY = 3
WORKERS     = 5  # 结构化回退并发数（只对需要回退的图片调用 API）

PROMPT = """检测并识别图片中的文字，将文本坐标格式化输出。"""

STRUCTURED_PROMPT = """You are extracting fields from a Zimbabwe ID card.

Return ONLY one JSON object with exactly these keys:
id_number, surname, first_name, birth_date, gender

Rules:
- id_number: 8 digits + 1 uppercase letter + 2 digits. Remove spaces, hyphens, and "CIT" suffix.
- surname: the family name. Ignore header/noise words like ZIMBABWE, NATIONAL, REGISTRATION, NATIONAL NUMBER, HO EO CO, HO ED CO, BIKITA, BEITBRIDGE, MBERENGWA, MAFANEDZA, TOMANI, SIKOBOKOBO.
- first_name: the given name(s). Keep spaces inside compound names.
- birth_date: normalize to DD.MM.YYYY. If 2-digit year, expand to 19xx or 20xx (year <= 30 → 20xx, else 19xx).
- gender: output "male" or "female". Look for M or F near "CIT" on the card.
- Do NOT guess. If you cannot see a field, return an empty string.

OCR line summary:
{ocr_summary}"""

# ── 正则 ──────────────────────────────────────────────────────
_ID_RE_PRIMARY   = re.compile(r"(\d{2})[\s\-.]*(\d{6,7})[\s\-.]*([A-Z0-9])[\s\-.]*(\d{2})")
_ID_RE_SECONDARY = re.compile(r"(\d{2})[\s\-.]*(\d{5})[\s\-.]*([A-Z0-9])[\s\-.]*(\d{2})")
_DATE_RE = re.compile(r"([0-9IJLO]{1,2})[\s./:\-]+([0-9IJLO]{1,2})[\s./:\-]+([0-9IJLO]{2,4})")
_DATE_COMPACT_RE = re.compile(r"\b([0-9IJLO]{2})([0-9IJLO]{2})([0-9IJLO]{4})\b")
_SEGMENT_RE = re.compile(r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)", re.DOTALL)
_CIT_LIKE_RE = re.compile(r"(?:CIT|CI1|C11|C1T|C1I|CII|GIT|IT)")
_GENDER_MARKER_RE = re.compile(r"(?:CIT|CI1|C11|C1T|C1I|CII|GIT|IT|ALIEN)")

_FIELDS = ("id_number", "surname", "first_name", "birth_date", "gender")

# ── 停用词 ────────────────────────────────────────────────────
_STOPWORDS = {
    "ZIMBABWE", "NATIONAL", "REGISTRATION", "NUMBER", "REPUBLIC", "IDENTITY",
    "CIT", "CITIZEN", "GIT", "NONE",
    "HO", "ED", "EO", "CO", "NO", "EU",
    "PV", "LTD", "ECOCASH", "CUSTOMER", "DECLARATION", "SIGNATURE", "OFFICIAL",
    "USE", "ONLY", "FORMER", "DATE", "ISSUE", "AGENTS", "PASSPORT",
    "CONNECTED", "TRENDS", "DIGITAL", "IAMDIGITAL", "NET", "WIRELESS", "FACEBOOK",
}
_PLACE_NAMES = {
    "BEITBRIDGE", "BULAWAYO", "MBERENGWA", "ZVISHAVANE", "BIKITA",
    "HARARE", "MUTARE", "MASVINGO", "GWERU", "KWEKWE", "CHITUNGWIZA",
    "EPWORTH", "MARONDERA", "BINDURA", "RUSAPE", "KADOMA", "SHURUGWI",
    "CHIREDZI", "KAROI", "VICTORIA FALLS", "HWANGE", "CHINHOYI",
    "MAFANEDZA", "SIKOBOKOBO", "MUSHAYI", "CHAKOTERA", "KENTARIBA",
    "RIVAHSE", "SARAI", "RUVENGO", "TOMANI", "GREEN SPUN",
    "HO EO CO", "HO ED CO", "NO ED CO",
}
_ALL_NOISE = _STOPWORDS | _PLACE_NAMES
_NAME_TOKEN_CORRECTIONS = {
    "DAFANA": "BAFANA",
}


def _clean_name_tokens(value: str, keep_short_inner: bool = False) -> list[str]:
    """清洗姓名候选 token，过滤页眉、地名、版面短噪声和过短 token。"""
    text = re.sub(r"[^A-Z ]", " ", value.upper())
    tokens = [t for t in re.split(r"\s+", text) if t]
    kept = []
    for pos, token in enumerate(tokens):
        token = _NAME_TOKEN_CORRECTIONS.get(token, token)
        if token in _ALL_NOISE:
            continue
        if len(token) < 3 and not (
            keep_short_inner
            and (
                (len(tokens) == 2 and len(tokens[0]) == 2 and len(tokens[1]) == 2)
                or (len(tokens) >= 3 and 0 < pos < len(tokens) - 1)
            )
        ):
            continue
        kept.append(token)
    return kept


def _clean_surname_value(value: str) -> str:
    """姓氏通常是单词；OCR 偶尔拆成 CHI ROMO，输出时合并为 CHIROMO。"""
    kept = _clean_name_tokens(value, keep_short_inner=True)
    return "".join(kept).strip()


def _clean_first_name_value(value: str) -> str:
    """名字可能是复合名，保留 token 间空格，避免 RODNEY JAMES 被合并。"""
    kept = _clean_name_tokens(value, keep_short_inner=True)
    if len(kept) >= 3 and any(len(token) < 3 for token in kept[1:]):
        return "".join(kept).strip()
    return " ".join(kept).strip()


def _normalize_id_candidate(value: str) -> str:
    clean = re.sub(r"[^A-Z0-9]", "", value.upper())
    clean = re.sub(r"CIT$", "", clean)
    if re.fullmatch(r"\d{8,9}[A-Z]\d{2}", clean):
        return clean
    m = re.fullmatch(r"(\d{8,9})(\d)(\d{2})", clean)
    if not m:
        return ""
    digit_to_letter = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "3": "E",
        "4": "A",
        "5": "S",
        # In this metal-ID batch the letter slot is often read as 6 when the
        # printed glyph is S, e.g. 04-0806546 04-CIT -> 04080654S04.
        "6": "S",
        "7": "T",
        "8": "B",
        "9": "P",
    }
    letter = digit_to_letter.get(m.group(2))
    return f"{m.group(1)}{letter}{m.group(3)}" if letter else ""


def _normalize_date_part(value: str) -> str:
    return value.upper().translate(str.maketrans({
        "I": "1",
        "J": "1",
        "L": "1",
        "O": "0",
    }))


def _prepare_image(image_path: Path) -> tuple[bytes, str]:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), "image/jpeg"


def _extract_segments(raw_text: str) -> list[dict]:
    segments = []
    for match in _SEGMENT_RE.finditer(raw_text):
        text, x1, y1, x2, y2 = match.groups()
        t = text.strip()
        if not t:
            continue
        segments.append({
            "text": t,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "cy": (int(y1) + int(y2)) // 2,
            "cx": (int(x1) + int(x2)) // 2,
        })
    return sorted(segments, key=lambda x: (x['cy'], x['cx']))


def _build_ocr_summary(raw_text: str, max_lines: int = 20) -> str:
    segments = _extract_segments(raw_text)
    lines = []
    for i, s in enumerate(segments, 1):
        t = s['text']
        if not t:
            continue
        lines.append(f"{i}. {t}")
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _extract_id_from_ocr(segments: list[dict], filename_stem: str | None = None) -> tuple[str, int | None]:
    candidates = []
    filename_id = _normalize_id_candidate(filename_stem or "")
    for idx, s in enumerate(segments):
        text = s['text'].upper().strip()
        if text in _STOPWORDS:
            continue
        compact_text = re.sub(r"[^A-Z0-9]", "", text)
        missing_letter_head = re.fullmatch(r"\d{8,9}", compact_text)
        if missing_letter_head and filename_id:
            for right_idx, right in enumerate(segments):
                if right_idx == idx:
                    continue
                if abs(right['cy'] - s['cy']) > 45:
                    continue
                if right['cx'] <= s['cx'] or right['cx'] - s['cx'] > 420:
                    continue
                right_compact = re.sub(r"[^A-Z0-9]", "", right['text'].upper().strip())
                suffix_match = re.match(r"(\d{2})", right_compact)
                if not suffix_match:
                    continue
                if not _GENDER_MARKER_RE.search(right_compact):
                    continue
                suffix = suffix_match.group(1)
                if filename_id.startswith(compact_text) and filename_id.endswith(suffix):
                    candidates.append((-165, s['cy'], filename_id, idx))
                    break
        split_head = re.fullmatch(r"(\d{8,9})([A-Z0-9])", compact_text)
        if split_head:
            for right_idx, right in enumerate(segments):
                if right_idx == idx:
                    continue
                if abs(right['cy'] - s['cy']) > 35:
                    continue
                if right['cx'] <= s['cx'] or right['cx'] - s['cx'] > 360:
                    continue
                right_text = right['text'].upper().strip()
                right_compact = re.sub(r"[^A-Z0-9]", "", right_text)
                suffix_match = re.match(r"(\d{2})", right_compact)
                if not suffix_match:
                    continue
                if not _GENDER_MARKER_RE.search(right_compact):
                    continue
                clean = _normalize_id_candidate(compact_text + suffix_match.group(1))
                if clean:
                    score = 180
                    if re.search(r"(?:CIT|CI1|C11|C1T|C1I|CII|GIT|IT|ALIEN)[MF]\b", right_compact):
                        score += 20
                    for near in segments:
                        if abs(near['cy'] - s['cy']) <= 35 and 0 < near['cx'] - right['cx'] <= 160:
                            if near['text'].upper().strip() in ("M", "F"):
                                score += 20
                                break
                    candidates.append((-score, s['cy'], clean, idx))
                    break
        if "CIT" in text:
            before_cit = text.split("CIT", 1)[0]
            compact = re.sub(r"[^A-Z0-9]", "", before_cit)
            clean = _normalize_id_candidate(compact)
            if clean:
                score = 140
                if idx > 0 and "NATIONAL NUMBER" in segments[idx - 1]['text'].upper():
                    score += 40
                candidates.append((-score, s['cy'], clean, idx))
                continue
        for pat in [_ID_RE_PRIMARY, _ID_RE_SECONDARY]:
            m = pat.search(text)
            if m:
                raw_id = f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}"
                clean = _normalize_id_candidate(raw_id)
                if clean:
                    score = 0
                    if "CIT" in text:
                        score += 100
                    if re.search(r"\d{2}[\s\-.]*\d{2}", text):
                        score += 20
                    if idx > 0 and "NATIONAL NUMBER" in segments[idx - 1]['text'].upper():
                        score += 40
                    # 主号码一般在姓名/生日上方；头像下方副编号位置更低，分数低。
                    candidates.append((-score, s['cy'], clean, idx))
                break
    if not candidates:
        return "", None
    candidates.sort()
    return candidates[0][2], candidates[0][3]


def _extract_gender_from_ocr(segments: list[dict], id_idx: int | None) -> str:
    for s in segments:
        text = s['text'].upper().strip()
        compact = re.sub(r"[^A-Z0-9]", "", text)
        if not _GENDER_MARKER_RE.search(compact):
            continue
        m = re.search(r"(?:CIT|CI1|C11|C1T|C1I|CII|GIT|IT|ALIEN)([MF])\b", compact)
        if m:
            return "male" if m.group(1) == "M" else "female"
        for near in segments:
            if abs(near['cy'] - s['cy']) > 35:
                continue
            if near['cx'] <= s['cx'] or near['cx'] - s['cx'] > 180:
                continue
            t = near['text'].upper().strip()
            if t in ("M", "F"):
                return "male" if t == "M" else "female"
    nearby_texts = []
    for offset in (-1, 0, 1):
        idx = (id_idx or 0) + offset
        if 0 <= idx < len(segments):
            nearby_texts.append(segments[idx]['text'])
    joined = " ".join(nearby_texts).upper()
    m = re.search(r'CIT\s*([MF])\b', joined)
    if m:
        return "male" if m.group(1) == "M" else "female"
    for s in segments:
        t = s['text'].upper().strip()
        if re.search(r'(?:CIT|CI1|C11|C1T|C1I|CII|GIT|IT|ALIEN)[\s]*F\b', t):
            return "female"
        if re.search(r'(?:CIT|CI1|C11|C1T|C1I|CII|GIT|IT|ALIEN)[\s]*M\b', t):
            return "male"
    if id_idx is not None:
        for offset in (-2, -1, 0, 1, 2):
            idx = id_idx + offset
            if 0 <= idx < len(segments):
                t = segments[idx]['text'].upper().strip()
                if t in ("M", "F"):
                    return "male" if t == "M" else "female"
    return ""


def _extract_date_from_ocr(segments: list[dict], id_idx: int | None = None) -> tuple[str, int | None]:
    dates = []
    id_y = segments[id_idx]['cy'] if id_idx is not None else None
    id_x = segments[id_idx]['cx'] if id_idx is not None else None
    for s in segments:
        m = _DATE_RE.search(s['text'])
        if m:
            d_raw, mo_raw, year_raw = m.group(1), m.group(2), m.group(3)
        else:
            m = _DATE_COMPACT_RE.search(s['text'])
            if not m:
                continue
            d_raw, mo_raw, year_raw = m.group(1), m.group(2), m.group(3)
        d_raw = _normalize_date_part(d_raw)
        mo_raw = _normalize_date_part(mo_raw)
        year_raw = _normalize_date_part(year_raw)
        try:
            d, mo = int(d_raw), int(mo_raw)
            if not (1 <= d <= 31 and 1 <= mo <= 12):
                continue
            if len(year_raw) == 4:
                y = int(year_raw)
                if not (1920 <= y <= 2020):
                    continue
            score = 0
            if id_y is not None:
                dy = s['cy'] - id_y
                if dy > 10:
                    score += 100
                    if dy <= 180:
                        score += 80
                    elif dy <= 280:
                        score += 20
                    else:
                        score -= 120
                else:
                    score -= 100
            if id_x is not None:
                score -= abs(s['cx'] - id_x) / 10
            if len(year_raw) == 4 and int(year_raw) >= 2000:
                score -= 80
            dates.append((-score, s['cy'], f"{str(d).zfill(2)}.{str(mo).zfill(2)}.{year_raw}", s['cy']))
        except ValueError:
            continue
    if dates:
        dates.sort()
        return dates[0][2], dates[0][3]
    return "", None


def _extract_names_from_ocr(
    segments: list[dict],
    id_y: int | None,
    date_y: int | None = None,
    id_x: int | None = None,
) -> tuple[str, str]:
    candidates = []
    for s in segments:
        t = s['text'].upper().strip()
        if not t:
            continue
        if id_y is not None and s['cy'] <= id_y + 10:
            continue
        if date_y is not None and s['cy'] >= date_y - 5:
            continue
        if id_x is not None and abs(s['cx'] - id_x) > 320:
            continue
        if t in _ALL_NOISE:
            continue
        raw_tokens = [tok for tok in re.split(r"[^A-Z]+", t) if tok]
        if any(tok in _STOPWORDS for tok in raw_tokens):
            continue
        if _DATE_RE.search(t):
            continue
        if _ID_RE_PRIMARY.search(t) or _ID_RE_SECONDARY.search(t):
            continue
        cleaned_name = _clean_first_name_value(t)
        letters = re.sub(r'[^A-Z]', '', cleaned_name)
        if len(letters) < 3:
            continue
        digits = sum(1 for c in t if c.isdigit())
        if digits > len(t) * 0.4:
            continue
        x_penalty = abs(s['cx'] - id_x) / 10 if id_x is not None else 0
        candidates.append((s['cy'], x_penalty, cleaned_name))
    if id_y is not None and candidates:
        below = [(c[0], c[1], c[2]) for c in candidates if c[0] > id_y + 10]
        if below:
            candidates = below
    deduped = []
    seen = set()
    candidates.sort(key=lambda item: (item[0], item[1]))
    for cy, _, text in candidates:
        key = text.upper().strip()
        if key not in seen:
            seen.add(key)
            deduped.append((cy, text))
    if len(deduped) >= 2:
        sn = _clean_surname_value(deduped[0][1])
        fn = _clean_first_name_value(deduped[1][1])
        return sn, fn
    elif len(deduped) == 1:
        sn = _clean_surname_value(deduped[0][1])
        return sn, ""
    return "", ""


def _needs_structured_fallback(result: dict) -> bool:
    if not result.get("id_number"):
        return True
    if not result.get("surname") or not result.get("first_name"):
        return True
    if not result.get("birth_date"):
        return True
    if result.get("gender") not in ("male", "female"):
        return True
    return False


def _call_structured_api(image_path: Path, raw_ocr: str) -> str:
    img_bytes, mime = _prepare_image(image_path)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    ocr_summary = _build_ocr_summary(raw_ocr)
    prompt = STRUCTURED_PROMPT.format(ocr_summary=ocr_summary)

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(f"{API_BASE_URL}/chat/completions",
                                 json=payload, headers=headers, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


def _extract_json_payload(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    return {
        "id_number": _normalize_id_candidate(str(data.get("id_number", ""))),
        "surname": _clean_surname_value(str(data.get("surname", ""))),
        "first_name": _clean_first_name_value(str(data.get("first_name", ""))),
        "birth_date": str(data.get("birth_date", "")).strip(),
        "gender": str(data.get("gender", "")).strip().lower(),
    }


def _merge_results(heuristic: dict, structured: dict | None) -> dict:
    if structured is None:
        return heuristic
    result = {}
    for field in _FIELDS:
        h_val = heuristic.get(field, "")
        s_val = structured.get(field, "")
        if field == "gender":
            result[field] = h_val if h_val in ("male", "female") else s_val
        elif field == "id_number":
            result[field] = h_val if h_val else s_val
        elif field in ("surname", "first_name"):
            result[field] = h_val if h_val else s_val
        elif field == "birth_date":
            result[field] = h_val if h_val else s_val
        else:
            result[field] = h_val or s_val
    return result


# ── 核心修改：返回结果和来源 ─────────────────────────────────
def process_single_image(image_path: Path) -> tuple[dict, dict]:
    """对单张图片进行两阶段字段提取，返回 (最终字段, 字段来源)"""
    ocr_file = OCR_DIR / f"{image_path.stem}.txt"
    if not ocr_file.exists():
        raise FileNotFoundError(f"OCR 结果缺失: {ocr_file}")

    raw_ocr = ocr_file.read_text(encoding="utf-8").strip()
    segments = _extract_segments(raw_ocr)

    # Phase 1: 规则提取
    id_number, id_idx = _extract_id_from_ocr(segments, image_path.stem)
    id_y = segments[id_idx]['cy'] if id_idx is not None else None
    id_x = segments[id_idx]['cx'] if id_idx is not None else None
    gender = _extract_gender_from_ocr(segments, id_idx)
    birth_date, birth_date_y = _extract_date_from_ocr(segments, id_idx)
    surname, first_name = _extract_names_from_ocr(segments, id_y, birth_date_y, id_x)

    result = {
        "id_number": id_number,
        "surname": surname,
        "first_name": first_name,
        "birth_date": birth_date,
        "gender": gender,
    }

    # 初始化来源：规则匹配到的字段记为 "heuristic"，否则 "none"
    sources = {}
    for k in _FIELDS:
        sources[k] = "heuristic" if result.get(k) else "none"

    # Phase 2: 结构化回退
    if _needs_structured_fallback(result):
        try:
            structured_text = _call_structured_api(image_path, raw_ocr)
            structured = _extract_json_payload(structured_text)
            if structured:
                # 合并结果，并标记哪些字段被结构化填补
                result = _merge_results(result, structured)
                for k in _FIELDS:
                    if sources[k] == "none" and result.get(k):
                        sources[k] = "structured"
        except Exception:
            pass

    # 确保所有字段都是字符串，空值即为 ""
    for k in _FIELDS:
        if result.get(k) is None:
            result[k] = ""
            sources[k] = "none"

    return result, sources


# ── 全局统计字典 ──────────────────────────────────────────────
field_stats = {field: {"heuristic": 0, "structured": 0, "none": 0} for field in _FIELDS}


def _process_one(img_p: Path) -> tuple[str, bool, str]:
    global field_stats
    try:
        result, sources = process_single_image(img_p)
        # 保存结果文件
        (DEAL_DIR / f"{img_p.stem}.txt").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        # 累加字段来源统计
        for field in _FIELDS:
            field_stats[field][sources.get(field, "none")] += 1
        return (img_p.name, True, "")
    except Exception as e:
        return (img_p.name, False, str(e))


def main():
    DEAL_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in IMAGE_DIR.iterdir()
                    if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    if MAX_COUNT > 0:
        images = images[:MAX_COUNT]

    total = len(images)
    pending = [p for p in images if not (DEAL_DIR / f"{p.stem}.txt").exists()]
    skipped = total - len(pending)
    if skipped:
        print(f"跳过已处理: {skipped} 张")

    if not pending:
        print("全部已完成。")
        return

    print(f"待处理: {len(pending)} 张  并发: {WORKERS} 线程\n")

    done = skipped
    fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process_one, p): p for p in pending}
        for future in as_completed(futures):
            name, ok, err = future.result()
            done += 1
            if ok:
                elapsed = time.time() - t0
                rate = (done - skipped) / elapsed * 60 if elapsed > 0 else 0
                print(f"[{done}/{total}] OK  {name}  ({rate:.0f}张/分)")
            else:
                fail += 1
                print(f"[{done}/{total}] FAIL {name}: {err}")

    elapsed = time.time() - t0
    print(f"\n完成  成功: {done - skipped - fail}  失败: {fail}  耗时: {elapsed:.0f}秒")

    # ── 输出字段来源统计报告 ─────────────────────────────────
    print("\n" + "=" * 60)
    print("字段提取来源统计")
    print("=" * 60)
    total_processed = done - skipped - fail  # 成功处理的图片数
    print(f"处理图片总数: {total_processed}")
    print(f"{'字段':<15} {'规则匹配':>8} {'结构化回退':>8} {'未提取':>8}")
    print("-" * 45)
    for field in _FIELDS:
        h = field_stats[field]["heuristic"]
        s = field_stats[field]["structured"]
        n = field_stats[field]["none"]
        print(f"{field:<15} {h:>8} ({h/total_processed*100:5.1f}%)  {s:>8} ({s/total_processed*100:5.1f}%)  {n:>8} ({n/total_processed*100:5.1f}%)")
    print("=" * 60)

    # （可选）将统计结果保存为 JSON 文件，方便后续分析
    stats_file = DEAL_DIR / "extraction_stats.json"
    stats_file.write_text(
        json.dumps({
            "total": total_processed,
            "fields": {f: dict(v) for f, v in field_stats.items()}
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"统计结果已保存至: {stats_file}")


if __name__ == "__main__":
    main()
