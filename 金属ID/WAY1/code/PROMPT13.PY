import base64
import io
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

# API 地址与鉴权
API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME   = "HunyuanOCR"
API_KEY      = "sk-placeholder-key"

# 待识别图片目录
IMAGE_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID")
# 识别结果输出目录（不存在会自动创建）
OUTPUT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY1\txt\金属ID-PROMPT13-OUT")

# 最多处理张数（0 = 处理全部）
MAX_COUNT   = 0
# 单次请求超时（秒）
TIMEOUT     = 120
# 失败自动重试次数（0 = 不重试）
MAX_RETRIES = 2
# 每次重试前等待时间（秒）
RETRY_DELAY = 3

# OCR 提示词 — 策略：
#   PROMPT2的结构约束（无示例值污染） + PROMPT12的性别平衡 + Python后处理做安全网
PROMPT = """Read this Zimbabwe ID card. Return ONLY a raw JSON object (no markdown, no explanation).

{
  "id_number": "",
  "surname": "",
  "first_name": "",
  "birth_date": "",
  "gender": ""
}

RULES:

1. id_number — The national ID at the top of the card.
   - Pattern: exactly 11 characters = 8 digits + 1 uppercase letter + 2 digits.
   - The card text may have spaces or dashes between groups. Read through them to find the 11-character ID.
   - Do NOT drop any character. If you see 11 meaningful characters, output all 11.

2. surname — The ALL-CAPS line directly below the id_number. Usually a single short word.
   - "ZIMBABWE" at the very top of the card is the country — never extract it.

3. first_name — The ALL-CAPS line directly below the surname. May be one or two words.

4. birth_date — Below the first_name. Output the date as printed on the card.

5. gender — Look for a single letter printed near the id_number area. Output the letter you see.
   - Read only the gender marker on this specific card.

OUTPUT: ONLY the raw JSON. No markdown. No explanation.
"""


# ══════════════════════════════════════════════════════════════
#  后处理（安全网：修复模型可能产生的格式问题）
# ══════════════════════════════════════════════════════════════

def _clean_id_number(raw: str) -> str:
    """清洗身份证号：去空格/横线/CIT，提取11位有效模式"""
    if not raw:
        return ""
    s = re.sub(r'\bCIT\b', '', raw, flags=re.IGNORECASE)
    s = re.sub(r'[^A-Za-z0-9]', '', s)
    # 提取：8数字 + 1字母 + 2数字
    m = re.search(r'(\d{8})([A-Za-z])(\d{2})', s)
    if m:
        return m.group(1) + m.group(2).upper() + m.group(3)
    # 备用：取前11个字母数字
    if len(s) >= 11:
        return s[:11].upper()
    return s.upper()


def _clean_name(raw: str) -> str:
    """清洗姓名：去ZIMBABWE，去首尾空白"""
    if not raw:
        return ""
    s = raw.strip()
    if s.upper() == "ZIMBABWE":
        return ""
    return s


def _clean_birth_date(raw: str) -> str:
    """标准化日期为 DD.MM.YYYY"""
    if not raw:
        return ""
    s = raw.strip()
    s = re.sub(r'[\s\-/]+', '.', s)
    parts = s.split('.')
    if len(parts) != 3:
        return raw.strip()
    day, month, year = [re.sub(r'\D', '', p) for p in parts]
    if not day or not month or not year:
        return raw.strip()
    day = day.zfill(2)
    month = month.zfill(2)
    if len(year) == 2:
        y = int(year)
        year = f"19{y:02d}" if y > 30 else f"20{y:02d}"
    return f"{day}.{month}.{year}"


def _clean_gender(raw: str) -> str:
    """标准化性别：F/female→female, M/male→male"""
    if not raw:
        return ""
    g = raw.strip().upper()
    if g in ("F", "FEMALE"):
        return "female"
    if g in ("M", "MALE"):
        return "male"
    return raw.strip()


def _post_process(raw_text: str) -> dict:
    """后处理流水线"""
    text = raw_text.strip()

    # 1. 去除 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)

    # 2. 解析 JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if not isinstance(data, dict):
        return {}

    # 3. 逐字段清洗
    return {
        "id_number":  _clean_id_number(data.get("id_number", "")),
        "surname":    _clean_name(data.get("surname", "")),
        "first_name": _clean_name(data.get("first_name", "")),
        "birth_date": _clean_birth_date(data.get("birth_date", "")),
        "gender":     _clean_gender(data.get("gender", "")),
    }


# ══════════════════════════════════════════════════════════════


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
    if not IMAGE_DIR.exists():
        print(f"[错误] 图片目录不存在: {IMAGE_DIR}")
        return
    if not IMAGE_DIR.is_dir():
        print(f"[错误] 路径不是目录: {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    width = len(str(total))

    print(f"图片目录 : {IMAGE_DIR}")
    print(f"输出目录 : {OUTPUT_DIR}")
    print(f"处理数量 : {total} 张  (MAX_COUNT={MAX_COUNT or '不限'})\n")

    success_count = 0
    skip_count    = 0
    fail_count    = 0
    postproc_fail = 0
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
            raw_result = ocr_with_retry(image_path)
            cleaned = _post_process(raw_result)
            if not cleaned:
                output_path.write_text(raw_result, encoding="utf-8")
                print(f"  → 后处理失败，保存原始输出: {output_path.name}")
                postproc_fail += 1
                continue
            output_path.write_text(
                json.dumps(cleaned, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  → 已保存: {output_path.name}")
            success_count += 1
        except Exception as exc:
            err_msg = str(exc)
            print(f"  → 失败: {err_msg}")
            fail_count += 1
            fail_list.append((image_path.name, err_msg))

    print(f"\n{'─' * 55}")
    print(f"处理完成  成功: {success_count}  跳过: {skip_count}  后处理失败: {postproc_fail}  请求失败: {fail_count}  共: {total}")
    if fail_list:
        print(f"\n失败文件列表（共 {len(fail_list)} 个）：")
        for name, err in fail_list:
            print(f"  ✗ {name}")
            print(f"      {err}")


if __name__ == "__main__":
    main()
