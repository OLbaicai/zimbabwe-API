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
DEAL_DIR   = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-4-DEAL-OUT")

MAX_COUNT   = 0
TIMEOUT     = 120
MAX_RETRIES = 2
RETRY_DELAY = 3
WORKERS     = 5

STRUCTURED_PROMPT = """Extract these 5 fields from this Zimbabwe ID card image:

Return ONLY a JSON object: {"id_number": "", "surname": "", "first_name": "", "birth_date": "", "gender": ""}

Rules:
- id_number: exactly 8 digits + 1 uppercase letter + 2 digits. Strip spaces/hyphens and trailing CIT/GIT/C1T.
- surname: family name only. Ignore headers, district/place names, addresses, signatures, dates and serial numbers.
- first_name: given name(s) after the surname. Do not return place names.
- birth_date: DD.MM.YYYY. Expand 2-digit year: <=30 -> 20xx, >30 -> 19xx.
- gender: "male" or "female". Look for M or F near CIT/GIT/C1T.
- Return "" for any field you cannot determine.

OCR hints, top-to-bottom:
{ocr_summary}"""

_ID_RE_PRIMARY   = re.compile(r"(\d{2})[\s\-.]*(\d{6,8})[\s\-.]*([A-Z])(?:[\s\-.]*)(\d{2})")
_ID_RE_SECONDARY = re.compile(r"(\d{2})[\s\-.]*(\d{5})[\s\-.]*([A-Z])(?:[\s\-.]*)(\d{2})")
_ID_RE_FALLBACK  = re.compile(r"(\d{2})\D+(\d{4,8})\D*([A-Z])\D*(\d{2})")
_DATE_RE = re.compile(r"(\d{1,2})[\s./\-](\d{1,2})[\s./\-](\d{2,4})")
_SEGMENT_RE = re.compile(r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)", re.DOTALL)
_CIT_MF_RE = re.compile(r"\b(?:CIT|GIT|C1T|ClT)\s*([MF])\b", re.IGNORECASE)
_LINE_TOLERANCE = 25
_FIELDS = ("id_number", "surname", "first_name", "birth_date", "gender")

_STOPWORDS = {
    "ZIMBABWE", "NATIONAL", "REGISTRATION", "NUMBER", "REPUBLIC", "IDENTITY",
    "CIT", "GIT", "C1T", "CITIZEN", "NONE", "HO", "ED", "EO", "CO", "NO",
    "DATE", "PLACE", "BIRTH", "SIGNATURE", "CARD", "IDENT", "ID",
}
_PLACE_NAMES = {
    "BEITBRIDGE", "BULAWAYO", "MBERENGWA", "ZVISHAVANE", "BIKITA",
    "HARARE", "MUTARE", "MASVINGO", "GWERU", "KWEKWE", "CHITUNGWIZA",
    "EPWORTH", "MARONDERA", "BINDURA", "RUSAPE", "KADOMA", "SHURUGWI",
    "CHIREDZI", "KAROI", "VICTORIA", "FALLS", "HWANGE", "CHINHOYI",
    "MAFANEDZA", "SIKOBOKOBO", "MUSHAYI", "CHAKOTERA", "KENTARIBA",
    "RIVAHSE", "NDUNA", "SARAI", "RUVENGO", "TOMANI", "GREEN", "SPUN",
    "MANICALAND", "MASHONALAND", "MATABELELAND", "MIDLANDS", "PLUMTREE",
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
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        segments.append({
            "text": t,
            "x": x1,
            "y": y1,
            "x2": x2,
            "y2": y2,
            "cy": (y1 + y2) // 2,
            "cx": (x1 + x2) // 2,
        })
    return sorted(segments, key=lambda x: (x["y"], x["x"]))


def _infer_page_size(segments: list[dict]) -> tuple[int, int]:
    if not segments:
        return 0, 0
    return max(int(s.get("x2", 0)) for s in segments), max(int(s.get("y2", 0)) for s in segments)


def _cluster_lines(segments: list[dict]) -> list[dict]:
    lines = []
    for s in segments:
        y, y2, x, x2 = int(s["y"]), int(s["y2"]), int(s["x"]), int(s["x2"])
        if not lines or y - int(lines[-1]["y_max"]) > _LINE_TOLERANCE:
            lines.append({"segments": [s], "y_min": y, "y_max": y, "y2_max": y2, "x_min": x, "x_max": x2})
        else:
            lines[-1]["segments"].append(s)
            lines[-1]["y_min"] = min(int(lines[-1]["y_min"]), y)
            lines[-1]["y_max"] = max(int(lines[-1]["y_max"]), y)
            lines[-1]["y2_max"] = max(int(lines[-1]["y2_max"]), y2)
            lines[-1]["x_min"] = min(int(lines[-1]["x_min"]), x)
            lines[-1]["x_max"] = max(int(lines[-1]["x_max"]), x2)

    for line in lines:
        segs = sorted(line["segments"], key=lambda item: int(item["x"]))
        line["segments"] = segs
        line["text"] = " ".join(str(item["text"]).strip() for item in segs if str(item["text"]).strip())
        line["cx"] = (int(line["x_min"]) + int(line["x_max"])) // 2
        line["cy"] = (int(line["y_min"]) + int(line["y2_max"])) // 2
    return lines


def _build_ocr_summary(raw_text: str, max_lines: int = 25) -> str:
    lines = _cluster_lines(_extract_segments(raw_text))
    out = []
    for i, line in enumerate(lines, 1):
        t = str(line.get("text", "")).strip()
        if t:
            out.append(f"{i}. {t}")
        if len(out) >= max_lines:
            break
    return "\n".join(out)


def _id_compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _normalize_id_piece(piece: str) -> str:
    normalized = []
    for i, ch in enumerate(piece.upper()):
        if i == len(piece) - 3:
            normalized.append(ch)
        elif ch in {"O", "Q", "C", "D"}:
            normalized.append("0")
        else:
            normalized.append(ch)
    return "".join(normalized)


def _valid_id(value: str) -> bool:
    return bool(re.fullmatch(r"\d{8,10}[A-Z]\d{2}", _id_compact(value)))


def _extract_id_from_line(text: str) -> str:
    compact = _id_compact(text.replace("CIT", "").replace("GIT", "").replace("C1T", ""))
    for length in (11, 12, 13):
        for start in range(0, max(len(compact) - length + 1, 0)):
            piece = compact[start:start + length]
            letter_pos = length - 3
            if not piece[letter_pos].isalpha():
                continue
            if not all(ch.isdigit() or ch in {"O", "Q", "C", "D"} for ch in piece[:letter_pos] + piece[letter_pos + 1:]):
                continue
            normalized = _normalize_id_piece(piece)
            if _valid_id(normalized):
                return normalized
    return ""


def _extract_id_from_ocr(segments: list[dict], lines: list[dict]) -> tuple[str, int | None, int | None]:
    candidates = []
    for idx, line in enumerate(lines):
        text = str(line.get("text", ""))
        line_id = _extract_id_from_line(text)
        if line_id:
            candidates.append((int(line.get("cy", 0)), line_id, idx, None))

    for idx, s in enumerate(segments):
        text = str(s["text"]).upper().strip()
        if text in _STOPWORDS:
            continue
        for pat in (_ID_RE_PRIMARY, _ID_RE_SECONDARY, _ID_RE_FALLBACK):
            m = pat.search(text)
            if not m:
                continue
            clean = _id_compact(f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}")
            if _valid_id(clean):
                candidates.append((int(s["cy"]), clean, None, idx))
            break

    if not candidates:
        return "", None, None
    candidates.sort(key=lambda item: (item[0], len(item[1])))
    _, value, line_idx, seg_idx = candidates[0]
    return value, line_idx, seg_idx


def _normalize_gender(value: str) -> str:
    v = value.strip().lower()
    if v in {"m", "male"}:
        return "male"
    if v in {"f", "female"}:
        return "female"
    return v


def _extract_gender_from_ocr(segments: list[dict], lines: list[dict], id_line_idx: int | None, id_seg_idx: int | None) -> str:
    for line in lines:
        m = _CIT_MF_RE.search(str(line.get("text", "")))
        if m:
            return "male" if m.group(1).upper() == "M" else "female"

    if id_line_idx is not None:
        joined = " ".join(str(lines[i].get("text", "")) for i in range(max(0, id_line_idx - 2), min(len(lines), id_line_idx + 3))).upper()
        m = _CIT_MF_RE.search(joined)
        if m:
            return "male" if m.group(1).upper() == "M" else "female"
        lone = re.search(r"\b([MF])\b", joined)
        if lone:
            return "male" if lone.group(1) == "M" else "female"

    if id_seg_idx is not None:
        for i in range(max(0, id_seg_idx - 2), min(len(segments), id_seg_idx + 3)):
            t = str(segments[i]["text"]).upper().strip()
            if t in {"M", "F"}:
                return "male" if t == "M" else "female"
    return ""


def _normalize_birth_date(date_text: str) -> str:
    m = _DATE_RE.search(date_text)
    if not m:
        return ""
    try:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    except ValueError:
        return ""
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return ""
    if len(m.group(3)) == 2:
        y = 1900 + y if y > 30 else 2000 + y
    if not (1920 <= y <= 2020):
        return ""
    return f"{d:02d}.{mo:02d}.{y}"


def _extract_date_from_ocr(lines: list[dict], id_line_idx: int | None) -> tuple[str, int | None]:
    candidates = []
    start = (id_line_idx + 1) if id_line_idx is not None else 0
    for idx, line in enumerate(lines[start:], start):
        text = str(line.get("text", ""))
        date = _normalize_birth_date(text)
        if not date:
            continue
        score = 100 - idx
        upper = text.upper()
        if id_line_idx is not None and idx > id_line_idx:
            score += 40
        if any(word in upper for word in ("BEITBRIDGE", "BIKITA", "MBERENGWA", "MAFANEDZA", "SIGN", "ISSUE")):
            score -= 30
        candidates.append((score, idx, date))
    if not candidates:
        return "", None
    candidates.sort(reverse=True)
    return candidates[0][2], candidates[0][1]


def _clean_name_line(text: str) -> str:
    tokens = []
    for token in re.split(r"\s+", text.strip()):
        cleaned = re.sub(r"[^A-Za-z]", "", token).upper()
        if not cleaned or cleaned in _ALL_NOISE:
            continue
        if len(cleaned) <= 2:
            continue
        tokens.append(cleaned)
    return " ".join(tokens).strip()


def _compact_split_name(name: str) -> str:
    tokens = [t for t in name.upper().split() if t]
    if len(tokens) != 2:
        return name
    if min(len(t) for t in tokens) <= 4 or sum(len(t) for t in tokens) >= 8:
        return "".join(tokens)
    return name


def _normalize_name(value: str) -> str:
    cleaned = _clean_name_line(value)
    if not cleaned:
        return ""
    return _compact_split_name(cleaned)


def _looks_like_name(value: str) -> bool:
    normalized = _normalize_name(value)
    if not normalized:
        return False
    if normalized in _ALL_NOISE:
        return False
    return len(re.sub(r"[^A-Z]", "", normalized)) >= 3


def _score_name_candidate(cleaned: str, raw: str, line: dict, page_w: int, page_h: int) -> int:
    joined = cleaned.replace(" ", "")
    if not joined:
        return -10000
    tokens = cleaned.split()
    score = len(joined)
    if len(tokens) == 1:
        score += 30
    elif len(tokens) == 2:
        score += 15
    else:
        score -= 15
    if any(t in _ALL_NOISE for t in tokens):
        score -= 60
    if any(len(t) <= 3 for t in tokens):
        score -= 10
    upper_raw = raw.upper()
    if re.search(r"\d", upper_raw):
        score -= 40
    if _DATE_RE.search(upper_raw):
        score -= 80
    cx, cy = int(line.get("cx", 0)), int(line.get("cy", 0))
    if page_w and int(page_w * 0.18) <= cx <= int(page_w * 0.82):
        score += 12
    if page_h and int(page_h * 0.20) <= cy <= int(page_h * 0.85):
        score += 8
    return score


def _pick_names_by_scoring(lines: list[dict], start: int, end: int, page_w: int, page_h: int) -> tuple[str, str]:
    candidates = []
    for idx in range(start, max(start, end)):
        raw = str(lines[idx].get("text", "")).strip()
        cleaned = _clean_name_line(raw)
        if not cleaned or _DATE_RE.search(raw):
            continue
        score = _score_name_candidate(cleaned, raw, lines[idx], page_w, page_h)
        candidates.append((score, idx, _normalize_name(cleaned)))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    picked = []
    seen = set()
    for score, idx, name in candidates:
        if score < 0 or not name or name in seen:
            continue
        seen.add(name)
        picked.append((idx, name))
        if len(picked) == 2:
            break
    picked.sort()
    if len(picked) >= 2:
        return picked[0][1], picked[1][1]
    if len(picked) == 1:
        return picked[0][1], ""
    return "", ""


def _pick_code_name_override(segments: list[dict], page_w: int, page_h: int) -> tuple[str, str]:
    short, long = [], []
    for s in segments:
        raw = str(s.get("text", "")).strip()
        if not raw or _DATE_RE.search(raw) or re.search(r"\d", raw):
            continue
        for part in re.split(r"\s+", raw):
            token = _normalize_name(part)
            if not token or token in _ALL_NOISE:
                continue
            item = {"t": token, "cx": int(s.get("cx", 0)), "cy": int(s.get("cy", 0))}
            if 3 <= len(token) <= 4:
                short.append(item)
            elif len(token) >= 6:
                long.append(item)
    best = None
    for s in short:
        for l in long:
            dx, dy = abs(s["cx"] - l["cx"]), abs(s["cy"] - l["cy"])
            if dx > 220 or dy > 260:
                continue
            score = 60 - min(30, dx // 10) - min(30, dy // 10)
            if page_w and s["cx"] <= page_w * 0.7 and l["cx"] <= page_w * 0.7:
                score += 10
            if best is None or score > best[0]:
                best = (score, s["t"], l["t"])
    return (best[1], best[2]) if best else ("", "")


def _extract_names_from_ocr(segments: list[dict], lines: list[dict], id_line_idx: int | None, date_line_idx: int | None) -> tuple[str, str]:
    page_w, page_h = _infer_page_size(segments)
    if id_line_idx is not None:
        start = id_line_idx + 1
        end = date_line_idx if date_line_idx is not None and date_line_idx > start else min(len(lines), start + 6)
        surname, first_name = _pick_names_by_scoring(lines, start, end, page_w, page_h)
    else:
        surname, first_name = _pick_names_by_scoring(lines, 0, min(len(lines), 10), page_w, page_h)

    code_surname, code_first = _pick_code_name_override(segments, page_w, page_h)
    if code_surname and code_first and (not surname or len(surname) > 4):
        surname, first_name = code_surname, code_first
    return surname, first_name


def _needs_structured_fallback(result: dict) -> bool:
    if not _valid_id(result.get("id_number", "")):
        return True
    if not _looks_like_name(result.get("surname", "")) or not _looks_like_name(result.get("first_name", "")):
        return True
    if not result.get("birth_date"):
        return True
    if result.get("gender") not in ("male", "female"):
        return True
    return False


def _call_structured_api(image_path: Path, raw_ocr: str) -> str:
    img_bytes, mime = _prepare_image(image_path)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = STRUCTURED_PROMPT.format(ocr_summary=_build_ocr_summary(raw_ocr))
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
            resp = requests.post(f"{API_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=TIMEOUT)
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
        if lines and lines[0].startswith("```"):
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
        "id_number": _id_compact(str(data.get("id_number", ""))),
        "surname": _normalize_name(str(data.get("surname", ""))),
        "first_name": _normalize_name(str(data.get("first_name", ""))),
        "birth_date": _normalize_birth_date(str(data.get("birth_date", ""))),
        "gender": _normalize_gender(str(data.get("gender", ""))),
    }


def _merge_results(heuristic: dict, structured: dict | None) -> dict:
    if structured is None:
        return heuristic

    h_id, s_id = heuristic.get("id_number", ""), structured.get("id_number", "")
    h_gen, s_gen = heuristic.get("gender", ""), structured.get("gender", "")
    result = {
        "id_number": h_id if _valid_id(h_id) else (s_id if _valid_id(s_id) else h_id),
        "surname": "",
        "first_name": "",
        "birth_date": heuristic.get("birth_date", "") or structured.get("birth_date", ""),
        "gender": h_gen if h_gen in ("male", "female") else s_gen,
    }

    for field in ("surname", "first_name"):
        h_val, s_val = heuristic.get(field, ""), structured.get(field, "")
        if _looks_like_name(h_val) and h_val not in _ALL_NOISE:
            result[field] = h_val
        elif _looks_like_name(s_val):
            result[field] = s_val
        else:
            result[field] = h_val or s_val
    return result


def process_single_image(image_path: Path) -> dict:
    ocr_file = OCR_DIR / f"{image_path.stem}.txt"
    if not ocr_file.exists():
        raise FileNotFoundError(f"OCR 结果缺失: {ocr_file}")

    raw_ocr = ocr_file.read_text(encoding="utf-8").strip()
    segments = _extract_segments(raw_ocr)
    lines = _cluster_lines(segments)

    id_number, id_line_idx, id_seg_idx = _extract_id_from_ocr(segments, lines)
    gender = _extract_gender_from_ocr(segments, lines, id_line_idx, id_seg_idx)
    birth_date, date_line_idx = _extract_date_from_ocr(lines, id_line_idx)
    surname, first_name = _extract_names_from_ocr(segments, lines, id_line_idx, date_line_idx)

    result = {
        "id_number": id_number,
        "surname": surname,
        "first_name": first_name,
        "birth_date": birth_date,
        "gender": gender,
    }

    if _needs_structured_fallback(result):
        try:
            result = _merge_results(result, _extract_json_payload(_call_structured_api(image_path, raw_ocr)))
        except Exception:
            pass

    return result


def _process_one(img_p: Path) -> tuple[str, bool, str]:
    try:
        result = process_single_image(img_p)
        (DEAL_DIR / f"{img_p.stem}.txt").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return (img_p.name, True, "")
    except Exception as e:
        return (img_p.name, False, str(e))


def main():
    DEAL_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    if MAX_COUNT > 0:
        images = images[:MAX_COUNT]

    total = len(images)
    print(f"待处理: {total} 张  并发: {WORKERS} 线程\n")
    done = 0
    fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process_one, p): p for p in images}
        for future in as_completed(futures):
            name, ok, err = future.result()
            done += 1
            if ok:
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                print(f"[{done}/{total}] OK  {name}  ({rate:.0f}张/分)")
            else:
                fail += 1
                print(f"[{done}/{total}] FAIL {name}: {err}")

    elapsed = time.time() - t0
    print(f"\n完成  成功: {done - fail}  失败: {fail}  耗时: {elapsed:.0f}秒")


if __name__ == "__main__":
    main()
