import base64
import io
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME   = "HunyuanOCR"
API_KEY      = "sk-placeholder-key"

IMAGE_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\身份证ID-金属\金属ID")
OUTPUT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\身份证ID-金属\金属ID-PROMPT6-OUT")

MAX_COUNT   = 0
TIMEOUT     = 120
MAX_RETRIES = 2
RETRY_DELAY = 3

# OCR 提示词 --- 策略：关键字段优先，强调id_number是所有字段中最重要的
PROMPT = """从津巴布韦身份证中提取信息。注意：id_number（身份证号）是最重要的字段，它的准确度直接影响整体结果。请把80%的注意力放在id_number上。

【最重要的字段：id_number】
- 位于卡片最上方，样子类似"02 033711 E 02"（带空格分段显示）
- 你需要在脑中执行这个清理过程：
  1. 看到 "02 033711 E 02" → 取所有数字和字母 → 02,033711,E,02
  2. 去掉空格 → 02033711E02
  3. 去掉末尾可能出现的"CIT"、"C11M"、"C1TF"等标记文字
  4. 输出干净的11位（8位数字+1个大写字母+2位数字）
- 如果第一个字符是0，请不要丢掉它！

【其余字段】
- surname: id_number正下方第一行
- first_name: surname正下方第一行
- birth_date: first_name下方，输出DD.MM.YYYY格式
- gender: 卡片上M或F的标记，M→male F→female

输出JSON（无markdown）：
{"id_number":"","surname":"","first_name":"","birth_date":"","gender":""}
"""

def _prepare_image(image_path: Path) -> tuple[bytes, str]:
    suffix = image_path.suffix.lower()
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    fmt  = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
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
    img_bytes, mime = _prepare_image(image_path)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"API 返回格式异常: {e}，原始响应: {data}") from e

def ocr_with_retry(image_path: Path) -> str:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return _call_api(image_path)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES + 1:
                time.sleep(RETRY_DELAY)
    raise last_exc

def main() -> None:
    if not IMAGE_DIR.exists():
        print(f"[错误] 图片目录不存在: {IMAGE_DIR}")
        return
    if not IMAGE_DIR.is_dir():
        print(f"[错误] 路径不是目录: {IMAGE_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in valid_exts)
    if not images:
        print(f"[警告] 图片目录中未找到 jpg/jpeg/png 文件: {IMAGE_DIR}")
        return
    if MAX_COUNT > 0:
        images = images[:MAX_COUNT]
    total = len(images)
    width = len(str(total))
    print(f"图片目录 : {IMAGE_DIR}")
    print(f"输出目录 : {OUTPUT_DIR}")
    print(f"处理数量 : {total} 张  (MAX_COUNT={MAX_COUNT or '不限'})\n")
    success_count = 0
    skip_count = 0
    fail_count = 0
    fail_list: list[tuple[str, str]] = []
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
    print(f"\n{'─' * 55}")
    print(f"处理完成  成功: {success_count}  跳过: {skip_count}  失败: {fail_count}  共: {total}")
    if fail_list:
        print(f"\n失败文件列表（共 {len(fail_list)} 个）：")
        for name, err in fail_list:
            print(f"  ✗ {name}")
            print(f"      {err}")

if __name__ == "__main__":
    main()
