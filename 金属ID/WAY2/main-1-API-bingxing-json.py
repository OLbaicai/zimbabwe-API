import base64
import io
import json
import logging
import re
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageOps

# ── 日志配置 ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ZimbabweID")

# ── 配置 ──────────────────────────────────────────────
API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME   = "HunyuanOCR"
API_KEY      = "sk-placeholder-key"

TIMEOUT     = 120
MAX_RETRIES = 2
RETRY_DELAY = 3

# 并发处理的最大线程数
MAX_CONCURRENT_WORKERS = 20  # 可根据需要调整

# ── Prompt 定义 ───────────────────────────────────────
PROMPT = "检测并识别图片中的文字，将文本坐标格式化输出。"

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

# ── 正则（不变） ────────────────────────────────────────
_ID_RE_PRIMARY   = re.compile(r"(\d{2})[\s\-.]*(\d{6,7})[\s\-.]*([A-Z])[\s\-.]*(\d{2})")
_ID_RE_SECONDARY = re.compile(r"(\d{2})[\s\-.]*(\d{5})[\s\-.]*([A-Z])[\s\-.]*(\d{2})")
_ID_RE_FALLBACK  = re.compile(r"(\d{2})\D+(\d{4,7})\D*([A-Z])\D*(\d{2})")
_DATE_RE = re.compile(r"(\d{1,2})[\s./\-](\d{1,2})[\s./\-](\d{2,4})")
_SEGMENT_RE = re.compile(r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)", re.DOTALL)

_FIELDS = ("id_number", "surname", "first_name", "birth_date", "gender")

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

# ── 线程池 ────────────────────────────────────────────
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)

# ── Pydantic 请求模型 ─────────────────────────────────
class ImageRequest(BaseModel):
    image_base64: str   # base64编码的图片数据

# ── 图像预处理 ──────────────────────────────────────────
def _prepare_image(image: Image.Image) -> tuple[bytes, str]:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), "image/jpeg"

# ── OCR 调用 (增强错误日志) ─────────────────────────────
def _call_ocr(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPT}
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 2048
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    last_exception = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(f"OCR 调用开始 (尝试 {attempt+1}/{MAX_RETRIES+1})")
            resp = requests.post(
                f"{API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=TIMEOUT
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info("OCR 调用成功")
            return content
        except Exception as e:
            last_exception = e
            logger.warning(f"OCR 调用失败 (尝试 {attempt+1}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"OCR 调用最终失败: {e}")
    raise RuntimeError(f"OCR 调用失败: {last_exception}")

# ── 结构化提取 API 调用 (增强错误日志) ─────────────────
def _call_structured_api(image_bytes: bytes, mime: str, raw_ocr: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    ocr_summary = _build_ocr_summary(raw_ocr)
    prompt = STRUCTURED_PROMPT.format(ocr_summary=ocr_summary)

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    last_exception = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(f"结构化提取调用开始 (尝试 {attempt+1})")
            resp = requests.post(
                f"{API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=TIMEOUT
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info("结构化提取调用成功")
            return content
        except Exception as e:
            last_exception = e
            logger.warning(f"结构化提取调用失败 (尝试 {attempt+1}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"结构化提取调用最终失败: {e}")
    raise RuntimeError(f"结构化提取调用失败: {last_exception}")

# ── 辅助函数（完全不变） ──────────────────────────────
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
        if re.search(r'CIT[\s]*F\b', t):
            return "female"
        if re.search(r'CIT[\s]*M\b', t):
            return "male"
    if id_idx is not None:
        for offset in (-2, -1, 0, 1, 2):
            idx = id_idx + offset
            if 0 <= idx < len(segments):
                t = segments[idx]['text'].upper().strip()
                if t in ("M", "F"):
                    return "male" if t == "M" else "female"
    return ""

def _extract_date_from_ocr(segments: list[dict]) -> str:
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
    if not result.get("id_number"):
        return True
    if not result.get("surname") or not result.get("first_name"):
        return True
    if not result.get("birth_date"):
        return True
    if result.get("gender") not in ("male", "female"):
        return True
    return False

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
        "id_number": str(data.get("id_number", "")).strip(),
        "surname": str(data.get("surname", "")).strip().upper(),
        "first_name": str(data.get("first_name", "")).strip().upper(),
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

# ── 核心处理函数（不变） ─────────────────────────────
def extract_from_image(image_bytes: bytes) -> dict:
    logger.info("开始处理图片...")
    img = Image.open(io.BytesIO(image_bytes))
    jpeg_bytes, mime = _prepare_image(img)

    # 第一步：OCR
    try:
        raw_ocr = _call_ocr(jpeg_bytes, mime)
    except Exception as e:
        logger.error(f"OCR 阶段失败: {e}")
        raise RuntimeError(f"OCR 处理失败: {e}")

    # 规则提取
    segments = _extract_segments(raw_ocr)
    id_number, id_idx = _extract_id_from_ocr(segments)
    id_y = segments[id_idx]['cy'] if id_idx is not None else None
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

    # 结构化回退
    if _needs_structured_fallback(result):
        logger.info("规则提取不完整，尝试结构化回退...")
        try:
            structured_text = _call_structured_api(jpeg_bytes, mime, raw_ocr)
            structured = _extract_json_payload(structured_text)
            result = _merge_results(result, structured)
        except Exception as e:
            logger.warning(f"结构化回退失败，使用规则结果: {e}")

    # 保证字段非 None
    for k in _FIELDS:
        if result.get(k) is None:
            result[k] = ""

    logger.info(f"提取完成: {result}")
    return result

# ── FastAPI 应用 ──────────────────────────────────────
app = FastAPI(title="Zimbabwe ID Card Extractor (JSON API)")

@app.on_event("startup")
async def startup_event():
    """启动时检测模型服务连通性（可选）"""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10
        )
        if resp.status_code == 200:
            logger.info("模型服务连通性检测成功")
        else:
            logger.warning(f"模型服务返回非200状态码: {resp.status_code}")
    except Exception as e:
        logger.error(f"模型服务连通性检测失败: {e}")
        logger.error("请确保模型服务已启动，且 API_BASE_URL 配置正确")

@app.post("/extract")
async def extract_id_fields(req: ImageRequest):
    """
    接收 JSON，包含 image_base64 字段，返回提取的五个字段。
    """
    try:
        # 解码 base64 得到图片字节
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 base64 图片数据")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, extract_from_image, image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("处理请求时发生异常")
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.on_event("shutdown")
def shutdown_event():
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4560)