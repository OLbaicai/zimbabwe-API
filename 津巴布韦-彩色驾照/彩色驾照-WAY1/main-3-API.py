import base64
import io
import json
import logging
import os
import re
import time
import zipfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageOps

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME   = "HunyuanOCR"
API_KEY      = "sk-placeholder-key"

TIMEOUT     = 120
MAX_RETRIES = 2
RETRY_DELAY = 3
MAX_CONCURRENT_WORKERS = 10   # 线程池大小

# 允许的图片后缀（小写）
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# 提示词（原版中文提示词，针对津巴布韦驾照）
PROMPT = """请从这张津巴布韦彩色驾照图片中提取以下字段，以严格 JSON 格式返回，不要包含任何其他内容：

【总体要求】
- 图片可能模糊或像素较低，请**尽最大努力**仔细观察每个文字区域，**不要轻易输出空字符串**。
- 即使文字很小、不清晰，也请尝试识别其内容，不要因为“看不清”就直接留空。
- 如果实在无法辨认任何字符，才允许输出空字符串 ""。

【字段1：id_number】
- 一般由 2位数字 + / + 8 位数字 + 1 位大写字母 + 2 位数字组成，例如：01/02033711E02。
- **特别注意：第 11 位一定是大写字母（A-Z）**，即使视觉上像数字也要纠正为字母：
  - 0 → O, 1 → I, 2 → Z, 5 → S, 8 → B 等，请结合上下文判断。
- 总共输出 13 或 14 位字符，不要增减。移除所有空格、横线、斜杠等分隔符。

【字段2：surname】
- 位置在 "surname" 标签右方，通常是**大写的简短字母组合**，不会包含空格。
- 不要输出标签文字本身，只输出姓氏部分。

【字段3：first_name】
- 位置在 "Name" 标签右方，通常是**大写字母**，表示名字。

【字段4：Date of Birth】
- 位置在 "Date of Birth" 标签右方，常见格式为 DD.MM.YY、DD.MM.YYYY 或 D/M/YY 等。
- 如果年份只有两位，请保留原样，不要自行补全。

【重要约束】
- 只输出数字和大写英文字母，**不输出任何中文、符号、空格或横线**。
- JSON 对象中必须包含全部四个字段，即使某个字段无法识别也要输出空字符串 ""，但不能缺失字段。
- 不要输出 null、None 或其他占位符。
- 仅返回 JSON，不要附加任何解释。

JSON 模板：
{
    "id_number": "",
    "surname": "",
    "first_name": "",
    "birth_date": "",
}
"""

# ═══════════════════ 后处理（与之前完全一致） ═══════════════════
def _try_parse_json(text: str) -> dict | None:
    s = text.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start:end+1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None

def _remove_diacritics(s: str) -> str:
    replacements = {
        'Á': 'A', 'À': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
        'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
        'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'Ý': 'Y', 'Ñ': 'N', 'Ç': 'C',
    }
    upper = s.upper()
    for accented, plain in replacements.items():
        upper = upper.replace(accented, plain)
    cleaned = re.sub(r'[^A-Z ]', '', upper)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def _fix_id_if_no_letter(raw_id: str) -> str:
    clean = re.sub(r'[^A-Z0-9]', '', raw_id.upper())
    if re.search(r'[A-Z]', clean):
        return raw_id

    digit_map = {'0': 'D', '1': 'L', '2': 'Z', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
    if len(clean) == 13:
        idx = 10
    elif len(clean) == 14:
        idx = 11
    else:
        return raw_id

    target_char = clean[idx]
    if target_char in digit_map:
        new_char = digit_map[target_char]
        clean = clean[:idx] + new_char + clean[idx+1:]
        return clean
    return raw_id

def postprocess_result(raw_text: str) -> str:
    obj = _try_parse_json(raw_text)
    if obj is None:
        return raw_text

    id_raw = obj.get("id_number", "")
    if id_raw:
        obj["id_number"] = _fix_id_if_no_letter(id_raw)

    for field in ("surname", "first_name"):
        val = obj.get(field, "")
        if val:
            obj[field] = _remove_diacritics(val)

    for key in ("id_number", "surname", "first_name", "birth_date"):
        if key not in obj or obj[key] is None:
            obj[key] = ""

    return json.dumps(obj, ensure_ascii=False)

# ═══════════════════ 图像处理与模型调用 ═══════════════════
def _prepare_image_from_bytes(image_bytes: bytes) -> tuple[bytes, str]:
    """从字节流加载图片并转换为 JPEG 字节，返回 (bytes, mime)"""
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

def _call_ocr_api(image_bytes: bytes) -> str:
    """使用字节流调用 HunyuanOCR API，返回识别文本"""
    jpeg_bytes, mime = _prepare_image_from_bytes(image_bytes)
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"API 调用最终失败: {last_exc}")

def extract_from_image_bytes(image_bytes: bytes, filename: str = "unknown") -> dict:
    """从图片字节流提取字段，返回包含字段及文件名的字典"""
    raw_text = _call_ocr_api(image_bytes)
    final_json_str = postprocess_result(raw_text)
    try:
        result = json.loads(final_json_str)
    except Exception:
        result = {"raw_output": final_json_str}
    result["_filename"] = filename
    return result

# ═══════════════════ FastAPI 应用 ═══════════════════
app = FastAPI(title="津巴布韦驾照 OCR API", version="1.0")
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)
logger = logging.getLogger("ZimbabweID")
logging.basicConfig(level=logging.INFO)

class ImageBase64Request(BaseModel):
    image_base64: str

@app.on_event("startup")
async def startup():
    logger.info(f"OCR API 服务启动，线程池大小: {MAX_CONCURRENT_WORKERS}")

@app.on_event("shutdown")
def shutdown():
    executor.shutdown(wait=True)

# ---------- 单张图片接口 ----------
@app.post("/extract")
async def extract_single(req: ImageBase64Request):
    """接收 JSON {"image_base64": "..."}，返回识别结果"""
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 base64 图片数据")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, extract_from_image_bytes, image_bytes, "base64_image")
        result.pop("_filename", None)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("处理请求时发生异常")
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")

@app.post("/extract_upload")
async def extract_upload(file: UploadFile = File(...)):
    """接收单张图片文件上传，返回识别结果"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持上传图片文件")

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="无法读取上传文件")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, extract_from_image_bytes, image_bytes, file.filename)
        result.pop("_filename", None)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("处理请求时发生异常")
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")

# ---------- 多张图片批量上传（传统方式，逐个选择文件） ----------
@app.post("/extract_batch")
async def extract_batch(files: List[UploadFile] = File(...)):
    """
    接收多张图片文件上传，返回每张图片的识别结果。
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一张图片")

    async def process_file(file: UploadFile) -> dict:
        if not file.content_type or not file.content_type.startswith("image/"):
            return {"filename": file.filename, "error": "不支持的文件类型"}
        try:
            image_bytes = await file.read()
        except Exception:
            return {"filename": file.filename, "error": "文件读取失败"}

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(executor, extract_from_image_bytes, image_bytes, file.filename)
            return result
        except Exception as e:
            return {"filename": file.filename, "error": str(e)}

    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    return JSONResponse(content={"results": results})

# ---------- ZIP 压缩包上传（全新方式，支持文件夹） ----------
@app.post("/extract_zip")
async def extract_zip(file: UploadFile = File(...)):
    """
    上传一个 ZIP 压缩包，里面包含多张图片（支持任意目录结构）。
    服务端会解压并识别所有支持的图片文件，返回每个图片的识别结果。
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅支持 .zip 格式的压缩包")

    try:
        zip_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="无法读取上传文件")

    # 使用临时目录解压
    with TemporaryDirectory() as tmp_dir:
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                # 安全检查：防止 Zip Slip 攻击
                for member in zf.infolist():
                    # 跳过目录
                    if member.is_dir():
                        continue
                    # 检查文件后缀
                    if Path(member.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                        continue
                    # 解压到临时目录（保持原有目录结构，避免同名冲突）
                    zf.extract(member, tmp_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="无效的 ZIP 文件")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"解压失败: {str(e)}")

        # 收集所有图片文件的路径
        img_files = []
        for root, dirs, files in os.walk(tmp_dir):
            for fname in files:
                if Path(fname).suffix.lower() in ALLOWED_EXTENSIONS:
                    img_files.append(Path(root) / fname)

        if not img_files:
            raise HTTPException(status_code=400, detail="ZIP 文件中未找到支持的图片文件（jpg/jpeg/png）")

        # 并发处理所有图片
        async def process_one_image(img_path: Path) -> dict:
            try:
                image_bytes = img_path.read_bytes()
            except Exception:
                return {"filename": img_path.name, "error": "文件读取失败"}
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(executor, extract_from_image_bytes, image_bytes, img_path.name)
                return result
            except Exception as e:
                return {"filename": img_path.name, "error": str(e)}

        tasks = [process_one_image(p) for p in img_files]
        results = await asyncio.gather(*tasks)

    # 清理临时文件（with 语句自动完成）
    return JSONResponse(content={
        "total_files": len(img_files),
        "results": results
    })

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4560)