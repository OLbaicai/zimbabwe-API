import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ========= 路径 =========

TXT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-OUT")

IMAGE_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID")

OUTPUT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID-检测框检验")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= 正则 =========

PATTERN = re.compile(
    r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)",
    re.DOTALL
)

# ========= 坐标还原 =========

def denormalize(x, y, w, h):
    px = int(float(x) * w / 1000.0)
    py = int(float(y) * h / 1000.0)
    return px, py

# ========= 画框 =========

def draw_boxes(image, text):

    # 关键！！！
    # 必须和OCR前处理一致
    image = ImageOps.exif_transpose(image)

    draw_img = image.copy()

    draw = ImageDraw.Draw(draw_img)

    overlay = Image.new("RGBA", draw_img.size, (0,0,0,0))
    overlay_draw = ImageDraw.Draw(overlay)

    W, H = draw_img.size

    print(f"Image size: {W} x {H}")

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    matches = PATTERN.finditer(text)

    for match in matches:

        try:
            content = match.group(1).strip()

            x1 = int(match.group(2))
            y1 = int(match.group(3))
            x2 = int(match.group(4))
            y2 = int(match.group(5))

            # ===== 关键 =====
            # HunyuanOCR是0~1000归一化坐标
            px1, py1 = denormalize(x1, y1, W, H)
            px2, py2 = denormalize(x2, y2, W, H)

            color = (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(0, 255),
            )

            alpha_color = color + (40,)

            # 画框
            draw.rectangle(
                [px1, py1, px2, py2],
                outline=color,
                width=3
            )

            # 半透明区域
            overlay_draw.rectangle(
                [px1, py1, px2, py2],
                fill=alpha_color
            )

            # 文本背景
            text_y = max(0, py1 - 22)

            bbox = draw.textbbox((0,0), content, font=font)

            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            draw.rectangle(
                [px1, text_y, px1 + tw + 4, text_y + th + 4],
                fill=(255,255,255)
            )

            draw.text(
                (px1 + 2, text_y + 2),
                content,
                fill=color,
                font=font
            )

        except Exception as e:
            print("Draw error:", e)

    draw_img.paste(overlay, (0,0), overlay)

    return draw_img

# ========= 主流程 =========

def main():

    txt_files = list(TXT_DIR.glob("*.txt"))

    print(f"发现 {len(txt_files)} 个txt文件")

    for idx, txt_path in enumerate(txt_files, 1):

        stem = txt_path.stem

        image_path = None

        for ext in [".jpg", ".jpeg", ".png"]:

            p = IMAGE_DIR / f"{stem}{ext}"

            if p.exists():
                image_path = p
                break

        if image_path is None:
            print(f"[{idx}] 找不到图片: {stem}")
            continue

        try:

            raw_text = txt_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            image = Image.open(image_path)

            result = draw_boxes(image, raw_text)

            out_path = OUTPUT_DIR / image_path.name

            result.save(out_path)

            print(f"[{idx}] 保存成功: {out_path.name}")

        except Exception as e:
            print(f"[{idx}] 失败: {stem}")
            print(e)

if __name__ == "__main__":
    main()