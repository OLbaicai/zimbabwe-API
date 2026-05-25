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
OUTPUT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\身份证ID-金属\金属ID-PROMPT3-OUT")

# 最多处理张数（0 = 处理全部）
MAX_COUNT   = 0
# 单次请求超时（秒）
TIMEOUT     = 120
# 失败自动重试次数（0 = 不重试）
MAX_RETRIES = 2
# 每次重试前等待时间（秒）
RETRY_DELAY = 3

# OCR 提示词
PROMPT = """你是津巴布韦身份证OCR识别专家。请按以下步骤操作：

【第1步】先写出你在卡片上看到的全部文字内容（保留原始空格和符号）。

【第2步】从上一步的文字中，找到身份证ID号码。它位于卡片顶部，格式为两段数字夹一个大写字母再跟两段数字，样子类似"02 033711 E 02"。注意：
- 把ID号码的每个字符单独列出来，像这样：0,2,0,3,3,7,1,1,E,0,2
- 去掉所有空格、横线、CIT字样，只保留11个字符（8个数字+1个大写字母+2个数字）
- 拼接成干净格式，例如：02033711E02
- 确认是11位，不多不少

【第3步】识别个人信息区（ID号码正下方依次排列）：
- surname（姓）：ID号正下方第一行，全大写字母，例如NGULUBE
- first_name（名）：surname正下方一行，全大写字母，复合名保留中间空格，例如RODNEY JAMES
- birth_date（出生日期）：first_name下方，统一格式DD.MM.YYYY（点分隔，四位年份），例如12.01.1961
- gender（性别）：ID号右侧的字母M或F。看清楚！女性是F，男性是M。不确定就填空字符串

【输出】只输出JSON，不要markdown不要解释：
{"id_number":"","surname":"","first_name":"","birth_date":"","gender":""}
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
