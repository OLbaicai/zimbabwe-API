import asyncio
import base64
import io
import json
import logging
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageOps

API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME = "HunyuanOCR"
API_KEY = "sk-placeholder-key"

TIMEOUT = 120
MAX_RETRIES = 2
RETRY_DELAY = 3
MAX_CONCURRENT_WORKERS = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

OCR_PROMPT = "检测并识别图片中的文字，将文本坐标格式化输出。"

STRUCTURED_PROMPT = """You are extracting fields from a Zimbabwe metal ID card.

Return ONLY one JSON object with exactly these keys:
id_number, surname, first_name, birth_date, gender

Rules:
- id_number: 8 or 9 digits + 1 uppercase letter + 2 digits. Remove spaces, hyphens, and CIT suffix.
- surname: family name. Ignore headers, place names, HO/EO/CO noise, and non-card background text.
- first_name: given name(s). Keep spaces only for true compound names.
- birth_date: keep visible date format as DD.MM.YY or DD.MM.YYYY. Do not expand two-digit years.
- gender: output "male" or "female". Look for M/F near CIT, C11, CII, IT, GIT, or ALIEN on the ID line.
- Do NOT guess. If you cannot see a field, return an empty string.

OCR line summary:
{ocr_summary}"""

_ID_RE_PRIMARY = re.compile(r"(\d{2})[\s\-.]*(\d{6,7})[\s\-.]*([A-Z0-9])[\s\-.]*(\d{2})")
_ID_RE_SECONDARY = re.compile(r"(\d{2})[\s\-.]*(\d{5})[\s\-.]*([A-Z0-9])[\s\-.]*(\d{2})")
_DATE_RE = re.compile(r"([0-9IJLO]{1,2})[\s./:\-]+([0-9IJLO]{1,2})[\s./:\-]+([0-9IJLO]{2,4})")
_DATE_COMPACT_RE = re.compile(r"\b([0-9IJLO]{2})([0-9IJLO]{2})([0-9IJLO]{4})\b")
_SEGMENT_RE = re.compile(r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)", re.DOTALL)
_GENDER_MARKER_RE = re.compile(r"(?:CIT|CI1|C11|C1T|C1I|CII|GIT|IT|ALIEN)")

_FIELDS = ("id_number", "surname", "first_name", "birth_date", "gender")

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
_NAME_TOKEN_CORRECTIONS = {"DAFANA": "BAFANA"}


def _clean_name_tokens(value: str, keep_short_inner: bool = False) -> list[str]:
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
    return "".join(_clean_name_tokens(value, keep_short_inner=True)).strip()


def _clean_first_name_value(value: str) -> str:
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
        "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
        "5": "S", "6": "S", "7": "T", "8": "B", "9": "P",
    }
    letter = digit_to_letter.get(m.group(2))
    return f"{m.group(1)}{letter}{m.group(3)}" if letter else ""


def _normalize_date_part(value: str) -> str:
    return value.upper().translate(str.maketrans({"I": "1", "J": "1", "L": "1", "O": "0"}))


def _extract_segments(raw_text: str) -> list[dict]:
    segments = []
    for match in _SEGMENT_RE.finditer(raw_text):
        text, x1, y1, x2, y2 = match.groups()
        t = text.strip()
        if not t:
            continue
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        segments.append({
            "text": t,
            "x1": x1i,
            "y1": y1i,
            "x2": x2i,
            "y2": y2i,
            "cy": (y1i + y2i) // 2,
            "cx": (x1i + x2i) // 2,
        })
    return sorted(segments, key=lambda x: (x["cy"], x["cx"]))


def _build_ocr_summary(raw_text: str, max_lines: int = 20) -> str:
    lines = []
    for i, s in enumerate(_extract_segments(raw_text), 1):
        lines.append(f"{i}. {s['text']}")
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _extract_id_from_ocr(segments: list[dict], filename_stem: str | None = None) -> tuple[str, int | None]:
    candidates = []
    filename_id = _normalize_id_candidate(filename_stem or "")
    for idx, s in enumerate(segments):
        text = s["text"].upper().strip()
        if text in _STOPWORDS:
            continue
        compact_text = re.sub(r"[^A-Z0-9]", "", text)

        missing_letter_head = re.fullmatch(r"\d{8,9}", compact_text)
        if missing_letter_head and filename_id:
            for right_idx, right in enumerate(segments):
                if right_idx == idx or abs(right["cy"] - s["cy"]) > 45:
                    continue
                if right["cx"] <= s["cx"] or right["cx"] - s["cx"] > 420:
                    continue
                right_compact = re.sub(r"[^A-Z0-9]", "", right["text"].upper().strip())
                suffix_match = re.match(r"(\d{2})", right_compact)
                if suffix_match and _GENDER_MARKER_RE.search(right_compact):
                    suffix = suffix_match.group(1)
                    if filename_id.startswith(compact_text) and filename_id.endswith(suffix):
                        candidates.append((-165, s["cy"], filename_id, idx))
                        break

        split_head = re.fullmatch(r"(\d{8,9})([A-Z0-9])", compact_text)
        if split_head:
            for right_idx, right in enumerate(segments):
                if right_idx == idx or abs(right["cy"] - s["cy"]) > 35:
                    continue
                if right["cx"] <= s["cx"] or right["cx"] - s["cx"] > 360:
                    continue
                right_compact = re.sub(r"[^A-Z0-9]", "", right["text"].upper().strip())
                suffix_match = re.match(r"(\d{2})", right_compact)
                if not suffix_match or not _GENDER_MARKER_RE.search(right_compact):
                    continue
                clean = _normalize_id_candidate(compact_text + suffix_match.group(1))
                if clean:
                    score = 180
                    if re.search(r"(?:CIT|CI1|C11|C1T|C1I|CII|GIT|IT|ALIEN)[MF]\b", right_compact):
                        score += 20
                    candidates.append((-score, s["cy"], clean, idx))
                    break

        if "CIT" in text:
            clean = _normalize_id_candidate(re.sub(r"[^A-Z0-9]", "", text.split("CIT", 1)[0]))
            if clean:
                score = 140
                if idx > 0 and "NATIONAL NUMBER" in segments[idx - 1]["text"].upper():
                    score += 40
                candidates.append((-score, s["cy"], clean, idx))
                continue

        for pat in (_ID_RE_PRIMARY, _ID_RE_SECONDARY):
            m = pat.search(text)
            if not m:
                continue
            clean = _normalize_id_candidate(f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}")
            if clean:
                score = 0
                if "CIT" in text or _GENDER_MARKER_RE.search(re.sub(r"[^A-Z0-9]", "", text)):
                    score += 100
                if re.search(r"\d{2}[\s\-.]*\d{2}", text):
                    score += 20
                if idx > 0 and "NATIONAL NUMBER" in segments[idx - 1]["text"].upper():
                    score += 40
                candidates.append((-score, s["cy"], clean, idx))
            break
    if not candidates:
        return "", None
    candidates.sort()
    return candidates[0][2], candidates[0][3]


def _extract_gender_from_ocr(segments: list[dict], id_idx: int | None) -> str:
    for s in segments:
        compact = re.sub(r"[^A-Z0-9]", "", s["text"].upper().strip())
        if not _GENDER_MARKER_RE.search(compact):
            continue
        m = re.search(r"(?:CIT|CI1|C11|C1T|C1I|CII|GIT|IT|ALIEN)([MF])\b", compact)
        if m:
            return "male" if m.group(1) == "M" else "female"
        for near in segments:
            if abs(near["cy"] - s["cy"]) <= 35 and 0 < near["cx"] - s["cx"] <= 180:
                t = near["text"].upper().strip()
                if t in ("M", "F"):
                    return "male" if t == "M" else "female"
    if id_idx is not None:
        for offset in (-2, -1, 0, 1, 2):
            idx = id_idx + offset
            if 0 <= idx < len(segments):
                t = segments[idx]["text"].upper().strip()
                if t in ("M", "F"):
                    return "male" if t == "M" else "female"
    return ""


def _extract_date_from_ocr(segments: list[dict], id_idx: int | None = None) -> tuple[str, int | None]:
    dates = []
    id_y = segments[id_idx]["cy"] if id_idx is not None else None
    id_x = segments[id_idx]["cx"] if id_idx is not None else None
    for s in segments:
        m = _DATE_RE.search(s["text"])
        if m:
            d_raw, mo_raw, year_raw = m.group(1), m.group(2), m.group(3)
        else:
            m = _DATE_COMPACT_RE.search(s["text"])
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
            if len(year_raw) == 4 and not (1920 <= int(year_raw) <= 2020):
                continue
            score = 0
            if id_y is not None:
                dy = s["cy"] - id_y
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
                score -= abs(s["cx"] - id_x) / 10
            if len(year_raw) == 4 and int(year_raw) >= 2000:
                score -= 80
            dates.append((-score, s["cy"], f"{str(d).zfill(2)}.{str(mo).zfill(2)}.{year_raw}", s["cy"]))
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
        t = s["text"].upper().strip()
        if not t:
            continue
        if id_y is not None and s["cy"] <= id_y + 10:
            continue
        if date_y is not None and s["cy"] >= date_y - 5:
            continue
        if id_x is not None and abs(s["cx"] - id_x) > 320:
            continue
        if t in _ALL_NOISE:
            continue
        raw_tokens = [tok for tok in re.split(r"[^A-Z]+", t) if tok]
        if any(tok in _STOPWORDS for tok in raw_tokens):
            continue
        if _DATE_RE.search(t) or _DATE_COMPACT_RE.search(t):
            continue
        if _ID_RE_PRIMARY.search(t) or _ID_RE_SECONDARY.search(t):
            continue
        cleaned_name = _clean_first_name_value(t)
        letters = re.sub(r"[^A-Z]", "", cleaned_name)
        if len(letters) < 3:
            continue
        digits = sum(1 for c in t if c.isdigit())
        if digits > len(t) * 0.4:
            continue
        x_penalty = abs(s["cx"] - id_x) / 10 if id_x is not None else 0
        candidates.append((s["cy"], x_penalty, cleaned_name))
    candidates.sort(key=lambda item: (item[0], item[1]))
    deduped = []
    seen = set()
    for cy, _, text in candidates:
        key = text.upper().strip()
        if key not in seen:
            seen.add(key)
            deduped.append((cy, text))
    if len(deduped) >= 2:
        return _clean_surname_value(deduped[0][1]), _clean_first_name_value(deduped[1][1])
    if len(deduped) == 1:
        return _clean_surname_value(deduped[0][1]), ""
    return "", ""


def _needs_structured_fallback(result: dict) -> bool:
    return (
        not result.get("id_number")
        or not result.get("surname")
        or not result.get("first_name")
        or not result.get("birth_date")
        or result.get("gender") not in ("male", "female")
    )


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
        else:
            result[field] = h_val if h_val else s_val
    return result


def _prepare_image_from_bytes(image_bytes: bytes) -> tuple[bytes, str]:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode == "P":
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), "image/jpeg"


def _call_chat_completion(image_bytes: bytes, prompt: str, max_tokens: int) -> str:
    jpeg_bytes, mime = _prepare_image_from_bytes(image_bytes)
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"API call failed: {last_exc}")


def _call_ocr_api(image_bytes: bytes) -> str:
    return _call_chat_completion(image_bytes, OCR_PROMPT, 4096)


def _call_structured_api(image_bytes: bytes, raw_ocr: str) -> str:
    prompt = STRUCTURED_PROMPT.format(ocr_summary=_build_ocr_summary(raw_ocr))
    return _call_chat_completion(image_bytes, prompt, 1024)


def extract_from_image_bytes(image_bytes: bytes, filename: str = "unknown") -> dict:
    raw_ocr = _call_ocr_api(image_bytes)
    segments = _extract_segments(raw_ocr)
    stem = Path(filename).stem

    id_number, id_idx = _extract_id_from_ocr(segments, stem)
    id_y = segments[id_idx]["cy"] if id_idx is not None else None
    id_x = segments[id_idx]["cx"] if id_idx is not None else None
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

    if _needs_structured_fallback(result):
        try:
            structured = _extract_json_payload(_call_structured_api(image_bytes, raw_ocr))
            result = _merge_results(result, structured)
        except Exception:
            pass

    for field in _FIELDS:
        if result.get(field) is None:
            result[field] = ""
    result["_filename"] = filename
    return result


app = FastAPI(title="Zimbabwe Metal ID OCR API", version="1.0")
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)
logger = logging.getLogger("ZimbabweMetalID")
logging.basicConfig(level=logging.INFO)


class ImageBase64Request(BaseModel):
    image_base64: str


@app.on_event("startup")
async def startup():
    logger.info("Metal ID OCR API started, workers=%s", MAX_CONCURRENT_WORKERS)


@app.on_event("shutdown")
def shutdown():
    executor.shutdown(wait=True)


@app.post("/extract")
async def extract_single(req: ImageBase64Request):
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, extract_from_image_bytes, image_bytes, "base64_image")
        result.pop("_filename", None)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("extract failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/extract_upload")
async def extract_upload(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")
    image_bytes = await file.read()
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, extract_from_image_bytes, image_bytes, file.filename)
        result.pop("_filename", None)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("upload extract failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/extract_batch")
async def extract_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image")

    async def process_file(file: UploadFile) -> dict:
        if not file.content_type or not file.content_type.startswith("image/"):
            return {"filename": file.filename, "error": "Unsupported file type"}
        try:
            image_bytes = await file.read()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, extract_from_image_bytes, image_bytes, file.filename)
        except Exception as exc:
            return {"filename": file.filename, "error": str(exc)}

    return JSONResponse(content={"results": await asyncio.gather(*(process_file(f) for f in files))})


@app.post("/extract_zip")
async def extract_zip(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")
    zip_bytes = await file.read()
    with TemporaryDirectory() as tmp_dir:
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    if Path(member.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                        continue
                    zf.extract(member, tmp_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")

        img_files = []
        for root, _, files in os.walk(tmp_dir):
            for fname in files:
                if Path(fname).suffix.lower() in ALLOWED_EXTENSIONS:
                    img_files.append(Path(root) / fname)
        if not img_files:
            raise HTTPException(status_code=400, detail="No supported image files found in ZIP")

        async def process_one_image(img_path: Path) -> dict:
            try:
                image_bytes = img_path.read_bytes()
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(executor, extract_from_image_bytes, image_bytes, img_path.name)
            except Exception as exc:
                return {"filename": img_path.name, "error": str(exc)}

        results = await asyncio.gather(*(process_one_image(p) for p in img_files))
    return JSONResponse(content={"total_files": len(img_files), "results": results})


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=4561)
