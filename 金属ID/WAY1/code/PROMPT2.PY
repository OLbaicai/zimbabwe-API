import base64
import io
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

# API 地址与鉴权
API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME   = "HunyuanOCR"
API_KEY      = "sk-placeholder-key"

# 待识别图片目录
IMAGE_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\身份证ID-金属\金属ID")
# 识别结果输出目录（不存在会自动创建）
OUTPUT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\身份证ID-金属\金属ID-PROMPT2-OUT")

# 最多处理张数（0 = 处理全部）
MAX_COUNT   = 0
# 单次请求超时（秒）
TIMEOUT     = 120
# 失败自动重试次数（0 = 不重试）
MAX_RETRIES = 2
# 每次重试前等待时间（秒）
RETRY_DELAY = 3

# OCR 提示词
PROMPT = """Extract fields from this Zimbabwe ID card. Output ONLY a raw JSON object (no markdown, no ```json, no explanation).

{
  "id_number": "",
  "surname": "",
  "first_name": "",
  "birth_date": "",
  "gender": ""
}

STRICT EXTRACTION RULES:

1. id_number
   - Format: 8 digits + 1 uppercase letter + 2 digits (11 characters total)
   - Example valid: 02033711E02
   - The raw text on card may look like: "02 061724 D 02 CIT" → you MUST strip ALL spaces, dashes, hyphens, dots, "CIT" text, and any other non-alphanumeric characters
   - CAREFULLY identify only the 11-character ID pattern. Remove ALL surrounding noise.
   - If you see "03-041276J03C11M" → extract only "03041276J03" (first 8 digits + 1 letter + next 2 digits = 11 chars)

2. surname (Surname / Family name)
   - Located DIRECTLY BELOW the id_number
   - ALL UPPERCASE letters only
   - WARNING: The word "ZIMBABWE" at the top of the card is the COUNTRY NAME, NOT a surname. DO NOT extract "ZIMBABWE" as surname.

3. first_name (First name / Given name)
   - Located DIRECTLY BELOW the surname
   - ALL UPPERCASE letters
   - Keep spaces for compound names (e.g., "RODNEY JAMES", not "RODNEY")

4. birth_date (Date of Birth)
   - Located BELOW the first_name
   - Output format: DD.MM.YYYY (dots as separators, 4-digit year)
   - Example: "12.01.1961" not "12 01 1961" not "12/01/1961"
   - If year is 2 digits, expand to 4: "72"→"1972", "05"→"2005"

5. gender
   - Look for a single letter "M" or "F" near the id_number area (usually to the right)
   - "M" → output "male"
   - "F" → output "female"
   - Look carefully: F and M can look similar on worn cards.

OUTPUT CONSTRAINTS:
- ONLY the raw JSON object. No markdown blocks. No explanation text. No ```json fences.
- All values must be plain ASCII (letters, digits, dots only)
- If a field cannot be read, use empty string ""
"""

# ══════════════════════════════════════════════════════════════


def _prepare_image(image_path: Path) -> tuple[bytes, str]:
    """
    打开图片，修正 EXIF 旋转，转为 JPEG/PNG 字节流。
    返回 (图片字节, MIME类型)。
    """
    suffix = image_path.suffix.lower()
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)  # 修正手机拍摄的旋转信息

    fmt  = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"

    # JPEG 不支持透明通道，统一转 RGB
    if fmt == "JPEG" and img.mode != "RGB":
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
    img.save(buf, format=fmt, quality=95)
    return buf.getvalue(), mime


def _call_api(image_path: Path) -> str:
    """调用 HunyuanOCR API，返回识别文本（已 strip）。"""
    img_bytes, mime = _prepare_image(image_path)
    b64 = base64.b64encode(img_bytes).decode("utf-8")

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

    resp = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"API 返回格式异常: {e}，原始响应: {data}") from e


def ocr_with_retry(image_path: Path) -> str:
    """带重试的 OCR 调用，全部失败则抛出最后一次异常。"""
    last_exc = None
    attempts = MAX_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            return _call_api(image_path)
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(RETRY_DELAY)
    raise last_exc


def main() -> None:
    # ── 前置检查 ──────────────────────────────────────────────
    if not IMAGE_DIR.exists():
        print(f"[错误] 图片目录不存在: {IMAGE_DIR}")
        return
    if not IMAGE_DIR.is_dir():
        print(f"[错误] 路径不是目录: {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 收集图片 ──────────────────────────────────────────────
    valid_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    )

    if not images:
        print(f"[警告] 图片目录中未找到 jpg/jpeg/png 文件: {IMAGE_DIR}")
        return

    if MAX_COUNT > 0:
        images = images[:MAX_COUNT]

    total = len(images)
    width = len(str(total))  # 进度编号对齐宽度

    print(f"图片目录 : {IMAGE_DIR}")
    print(f"输出目录 : {OUTPUT_DIR}")
    print(f"处理数量 : {total} 张  (MAX_COUNT={MAX_COUNT or '不限'})\n")

    success_count = 0
    skip_count    = 0
    fail_count    = 0
    fail_list: list[tuple[str, str]] = []  # (文件名, 错误信息)

    # ── 逐张处理 ──────────────────────────────────────────────
    for idx, image_path in enumerate(images, 1):
        prefix = f"[{idx:{width}d}/{total}]"
        output_path = OUTPUT_DIR / (image_path.stem + ".txt")

        if output_path.exists():
            print(f"{prefix} 跳过（已存在）: {image_path.name}")
            skip_count += 1
            continue

        print(f"{prefix} 处理: {image_path.name}", end="", flush=True)

        try:
            result = ocr_with_retry(image_path)
            output_path.write_text(result, encoding="utf-8")
            print(f"  → 已保存: {output_path.name}")
            success_count += 1
        except Exception as exc:
            err_msg = str(exc)
            print(f"  → 失败: {err_msg}")
            fail_count += 1
            fail_list.append((image_path.name, err_msg))

    # ── 汇总报告 ──────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"处理完成  成功: {success_count}  跳过: {skip_count}  失败: {fail_count}  共: {total}")
    if fail_list:
        print(f"\n失败文件列表（共 {len(fail_list)} 个）：")
        for name, err in fail_list:
            print(f"  ✗ {name}")
            print(f"      {err}")


if __name__ == "__main__":
    main()
