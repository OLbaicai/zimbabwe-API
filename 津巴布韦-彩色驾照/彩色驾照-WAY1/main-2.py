# 相比main-1，这个版本增加了对变音字母的修正（例如 Á → A），以提高识别的鲁棒性。其他部分与main-1保持一致。
import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageOps

# ═══════════════════════════════════════════════════════════
# API 地址与鉴权
API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME   = "HunyuanOCR"
API_KEY      = "sk-placeholder-key"

# 待识别图片目录
IMAGE_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\津巴布韦-彩色驾照\彩色驾照图像")
# 识别结果输出目录（不存在会自动创建）
OUTPUT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\津巴布韦-彩色驾照\txt\MAIN-2-OUT")

# 最多处理张数（0 = 处理全部）
MAX_COUNT   = 0
# 单次请求超时（秒）
TIMEOUT     = 120
# 失败自动重试次数（0 = 不重试）
MAX_RETRIES = 2
# 每次重试前等待时间（秒）
RETRY_DELAY = 3
# 并发线程数
WORKERS     = 15

# OCR 提示词
PROMPT = """请从这张津巴布韦身份证图片中提取以下字段，以严格 JSON 格式返回，不要包含任何其他内容：

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

# ═══════════════════ 后处理（变音字母修正） ═══════════════════

def _try_parse_json(text: str) -> dict | None:
    """尝试从文本中提取JSON对象。"""
    s = text.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # 尝试提取第一个 { ... } 块
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
    """替换变音字母为普通大写字母，仅保留A-Z和空格。"""
    # 常见变音字母映射
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
    # 移除所有非A-Z非空格的字符，合并多余空格
    cleaned = re.sub(r'[^A-Z ]', '', upper)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def postprocess_result(raw_text: str) -> str:
    """
    对模型返回的JSON字符串进行后处理：
    - 修正surname和first_name中的变音字母
    - 保证JSON结构完整，缺失字段补空字符串
    - 若无法解析为JSON，保留原始输出
    """
    obj = _try_parse_json(raw_text)
    if obj is None:
        return raw_text   # 保持原样

    # 处理姓名
    for field in ("surname", "first_name"):
        val = obj.get(field, "")
        if val:
            obj[field] = _remove_diacritics(val)

    # 确保四个字段都存在，且值为字符串（防止null）
    for key in ("id_number", "surname", "first_name", "birth_date"):
        if key not in obj or obj[key] is None:
            obj[key] = ""

    return json.dumps(obj, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════

def _prepare_image(image_path: Path) -> tuple[bytes, str]:
    """打开图片，修正EXIF旋转，转为JPEG/PNG字节流。"""
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


def _process_one(image_path: Path) -> tuple[str, bool, str]:
    try:
        raw_result = ocr_with_retry(image_path)
        # ---------- 后处理：修正变音字母 ----------
        final_result = postprocess_result(raw_result)
        output_path = OUTPUT_DIR / (image_path.stem + ".txt")
        output_path.write_text(final_result, encoding="utf-8")
        return (image_path.name, True, "")
    except Exception as e:
        return (image_path.name, False, str(e))


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
    pending = [p for p in images if not (OUTPUT_DIR / f"{p.stem}.txt").exists()]
    skipped = total - len(pending)

    width = len(str(total))

    print(f"图片目录 : {IMAGE_DIR}")
    print(f"输出目录 : {OUTPUT_DIR}")
    print(f"处理数量 : {total} 张  (MAX_COUNT={MAX_COUNT or '不限'})")
    if skipped:
        print(f"跳过已处理: {skipped} 张")
    print(f"待处理: {len(pending)} 张  并发: {WORKERS} 线程\n")

    if not pending:
        print("全部已完成。")
        return

    success_count = 0
    fail_count = 0
    fail_list: list[tuple[str, str]] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process_one, p): p for p in pending}
        for future in as_completed(futures):
            name, ok, err = future.result()
            if ok:
                success_count += 1
                elapsed = time.time() - t0
                rate = success_count / elapsed * 60 if elapsed > 0 else 0
                print(f"[{success_count + skipped:{width}d}/{total}] OK  {name}  ({rate:.0f}张/分)")
            else:
                fail_count += 1
                print(f"[{success_count + skipped + fail_count:{width}d}/{total}] FAIL {name}: {err}")
                fail_list.append((name, err))

    elapsed = time.time() - t0
    print(f"\n{'─' * 55}")
    print(f"处理完成  成功: {success_count}  跳过: {skipped}  失败: {fail_count}  共: {total}")
    print(f"耗时: {elapsed:.1f}秒")
    if fail_list:
        print(f"\n失败文件列表（共 {len(fail_list)} 个）：")
        for name, err in fail_list:
            print(f"  ✗ {name}")
            print(f"      {err}")


if __name__ == "__main__":
    main()