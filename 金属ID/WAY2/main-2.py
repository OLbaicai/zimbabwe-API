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

IMAGE_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID")
OCR_DIR    = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-OUT")
DEAL_DIR   = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-2-DEAL-OUT")

MAX_COUNT   = 0
TIMEOUT     = 120
MAX_RETRIES = 2
RETRY_DELAY = 3
WORKERS     = 5  # 并发数（仅对需要结构化回退的图片调 API）

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
_ID_RE_PRIMARY   = re.compile(r"(\d{2})[\s\-.]*(\d{6,7})[\s\-.]*([A-Z])[\s\-.]*(\d{2})")
_ID_RE_SECONDARY = re.compile(r"(\d{2})[\s\-.]*(\d{5})[\s\-.]*([A-Z])[\s\-.]*(\d{2})")
_ID_RE_FALLBACK  = re.compile(r"(\d{2})\D+(\d{4,7})\D*([A-Z])\D*(\d{2})")
_DATE_RE = re.compile(r"(\d{1,2})[\s./\-](\d{1,2})[\s./\-](\d{2,4})")
_SEGMENT_RE = re.compile(r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)", re.DOTALL)

_FIELDS = ("id_number", "surname", "first_name", "birth_date", "gender")

# ── 停用词 ────────────────────────────────────────────────────
_STOPWORDS = {
    "ZIMBABWE", "NATIONAL", "REGISTRATION", "NUMBER", "REPUBLIC", "IDENTITY",
    "CIT", "CITIZEN", "NONE",
}
_PLACE_NAMES = {
    "BEITBRIDGE", "BULAWAYO", "MBERENGWA", "ZVISHAVANE", "BIKITA",
    "HARARE", "MUTARE", "MASVINGO", "GWERU", "KWEKWE", "CHITUNGWIZA",
    "EPWORTH", "MARONDERA", "BINDURA", "RUSAPE", "KADOMA", "SHURUGWI",
    "CHIREDZI", "KAROI", "VICTORIA FALLS", "HWANGE", "CHINHOYI",
    "MAFANEDZA", "SIKOBOKOBO", "MUSHAYI", "CHAKOTERA", "KENTARIBA",
    "RIVAHSE", "NDUNA", "SARAI", "RUVENGO", "TOMANI", "GREEN SPUN",
    "HO EO CO", "HO ED CO", "NO ED CO",
}
_ALL_NOISE = _STOPWORDS | _PLACE_NAMES


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
            "cy": (int(y1) + int(y2)) // 2,
            "cx": (int(x1) + int(x2)) // 2,
        })
    return sorted(segments, key=lambda x: (x['cy'], x['cx']))


def _build_ocr_summary(raw_text: str, max_lines: int = 20) -> str:
    """把原始 OCR 坐标串压缩成行号摘要，供结构化回退用。"""
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


def _extract_id_from_ocr(segments: list[dict]) -> tuple[str, int | None]:
    """多策略从 OCR 段提取 id_number，返回 (id_number, segment_index)"""
    candidates = []
    for idx, s in enumerate(segments):
        text = s['text'].upper().strip()
        if text in _STOPWORDS:
            continue
        for pat in [_ID_RE_PRIMARY, _ID_RE_SECONDARY, _ID_RE_FALLBACK]:
            m = pat.search(text)
            if m:
                raw_id = f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}"
                clean = re.sub(r'[^A-Z0-9]', '', raw_id)
                if 10 <= len(clean) <= 12:
                    candidates.append((s['cy'], clean, idx))
                break

    if not candidates:
        return "", None

    candidates.sort()
    return candidates[0][1], candidates[0][2]


def _extract_gender_from_ocr(segments: list[dict], id_idx: int | None) -> str:
    """从 OCR 段提取性别：检查 ID 段及其相邻段"""
    # 策略1: 拼接 ID 段 ±1 范围，搜索 CIT M/F
    nearby_texts = []
    for offset in (-1, 0, 1):
        idx = (id_idx or 0) + offset
        if 0 <= idx < len(segments):
            nearby_texts.append(segments[idx]['text'])
    joined = " ".join(nearby_texts).upper()

    m = re.search(r'CIT\s*([MF])\b', joined)
    if m:
        return "male" if m.group(1) == "M" else "female"

    # 策略2: 全局搜索 CIT F / CIT M
    for s in segments:
        t = s['text'].upper().strip()
        if re.search(r'CIT[\s]*F\b', t):
            return "female"
        if re.search(r'CIT[\s]*M\b', t):
            return "male"

    # 策略3: ID 段附近的独立纯 M/F 标记
    if id_idx is not None:
        for offset in (-2, -1, 0, 1, 2):
            idx = id_idx + offset
            if 0 <= idx < len(segments):
                t = segments[idx]['text'].upper().strip()
                if t in ("M", "F"):
                    return "male" if t == "M" else "female"

    return ""


def _extract_date_from_ocr(segments: list[dict]) -> str:
    """从 OCR 段提取出生日期"""
    dates = []
    for s in segments:
        m = _DATE_RE.search(s['text'])
        if not m:
            continue
        try:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if not (1 <= d <= 31 and 1 <= mo <= 12):
                continue
            if len(m.group(3)) == 2:
                y = 1900 + y if y > 30 else 2000 + y
            if not (1920 <= y <= 2020):
                continue
            dates.append((s['cy'], f"{str(d).zfill(2)}.{str(mo).zfill(2)}.{y}"))
        except ValueError:
            continue
    if dates:
        dates.sort()
        return dates[0][1]
    return ""


def _extract_names_from_ocr(segments: list[dict], id_y: int | None) -> tuple[str, str]:
    """从坐标排序的 OCR 段提取 surname 和 first_name"""
    candidates = []
    for s in segments:
        t = s['text'].upper().strip()
        if not t:
            continue
        if t in _ALL_NOISE:
            continue
        if any(w in t for w in _STOPWORDS):
            continue
        if _DATE_RE.search(t):
            continue
        if _ID_RE_PRIMARY.search(t) or _ID_RE_SECONDARY.search(t):
            continue
        letters = re.sub(r'[^A-Z]', '', t)
        if len(letters) < 2:
            continue
        digits = sum(1 for c in t if c.isdigit())
        if digits > len(t) * 0.4:
            continue
        candidates.append((s['cy'], s['text']))

    if id_y is not None and candidates:
        below = [(c[0], c[1]) for c in candidates if c[0] > id_y + 10]
        if below:
            candidates = below

    deduped = []
    seen = set()
    for cy, text in candidates:
        key = text.upper().strip()
        if key not in seen:
            seen.add(key)
            deduped.append((cy, text))

    if len(deduped) >= 2:
        sn = re.sub(r'[^A-Z ]', '', deduped[0][1].upper()).strip()
        fn = re.sub(r'[^A-Z ]', '', deduped[1][1].upper()).strip()
        return sn, fn
    elif len(deduped) == 1:
        sn = re.sub(r'[^A-Z ]', '', deduped[0][1].upper()).strip()
        return sn, ""
    return "", ""


def _needs_structured_fallback(result: dict) -> bool:
    """判断是否需要调用结构化模型回退"""
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
    """基于图片 + OCR 摘要，调用模型生成结构化 JSON"""
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
    """从模型回复中提取 JSON 对象"""
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
        "id_number": str(data.get("id_number", "")).strip(),
        "surname": str(data.get("surname", "")).strip().upper(),
        "first_name": str(data.get("first_name", "")).strip().upper(),
        "birth_date": str(data.get("birth_date", "")).strip(),
        "gender": str(data.get("gender", "")).strip().lower(),
    }


def _merge_results(heuristic: dict, structured: dict | None) -> dict:
    """合并规则提取和结构化提取的结果，规则优先（坐标定位更可靠）"""
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


def process_single_image(image_path: Path) -> dict:
    """两阶段字段提取：规则提取 + 结构化 API 回退"""
    ocr_file = OCR_DIR / f"{image_path.stem}.txt"
    if not ocr_file.exists():
        raise FileNotFoundError(f"OCR 结果缺失: {ocr_file}")

    raw_ocr = ocr_file.read_text(encoding="utf-8").strip()
    segments = _extract_segments(raw_ocr)

    # ── Phase 1: 规则提取 ──
    id_number, id_idx = _extract_id_from_ocr(segments)

    id_y = None
    if id_idx is not None:
        id_y = segments[id_idx]['cy']

    gender = _extract_gender_from_ocr(segments, id_idx)
    birth_date = _extract_date_from_ocr(segments)
    surname, first_name = _extract_names_from_ocr(segments, id_y)

    result = {
        "id_number": id_number,
        "surname": surname,
        "first_name": first_name,
        "birth_date": birth_date,
        "gender": gender,
    }

    # ── Phase 2: 结构化 API 回退（仅当字段不完整时）──
    if _needs_structured_fallback(result):
        try:
            structured_text = _call_structured_api(image_path, raw_ocr)
            structured = _extract_json_payload(structured_text)
            result = _merge_results(result, structured)
        except Exception:
            pass

    return result


def _process_one(img_p: Path) -> tuple[str, bool, str]:
    try:
        result = process_single_image(img_p)
        (DEAL_DIR / f"{img_p.stem}.txt").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
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


if __name__ == "__main__":
    main()
