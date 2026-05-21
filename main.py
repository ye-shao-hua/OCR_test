import os
import sys
import time
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/simhei.ttf"


def preprocess_image(img_path):
    """轻量预处理：缩放 + 灰度 + CLAHE"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    max_side = 1600
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def draw_ocr_boxes(img_orig, detections, img_name, result_dir):
    """用 PIL 绘制检测框和中文标注，避免乱码"""
    h, w = img_orig.shape[:2]
    base = max(h, w)

    # PIL 图像转换（BGR → RGB）
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 自适应参数
    box_w = max(2, base // 900)
    font_size = max(12, base // 45)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    # 颜色方案（RGB）
    def get_color(conf):
        if conf >= 70:
            return (0, 180, 0, 220)
        elif conf >= 30:
            return (255, 140, 0, 220)
        else:
            return (220, 30, 30, 220)

    occupied = []

    def is_overlap(yt, yb):
        m = font_size
        for oy1, oy2 in occupied:
            if not (yb + m < oy1 or yt - m > oy2):
                return True
        return False

    for bbox, text, confidence in detections:
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        conf_pct = confidence * 100
        color = get_color(conf_pct)

        # 检测框
        draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=box_w)

        # 标签文字
        label = f"{text} ({conf_pct:.1f}%)"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]

        pad = font_size // 3
        bg_x1 = x1
        bg_y1 = y1 - th - pad
        bg_x2 = x1 + tw + pad * 2
        bg_y2 = y1

        # 上方空间不足则放框内上方
        if bg_y1 < 0 or is_overlap(bg_y1, bg_y2):
            bg_y1 = y1 + pad
            bg_y2 = y1 + th + pad * 2
            if bg_y2 > h:
                bg_y1 = y2 - th - pad
                bg_y2 = y2

        bg_x1 = max(0, bg_x1)
        bg_x2 = min(w, bg_x2)
        bg_y1 = max(0, bg_y1)
        bg_y2 = min(h, bg_y2)

        # 半透明背景
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2],
                               fill=color, outline=color[:3], width=box_w)
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(pil_img)

        # 文字（白色）
        text_x = bg_x1 + pad
        text_y = bg_y1 + (bg_y2 - bg_y1 - th) // 2
        draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))

        occupied.append((bg_y1, bg_y2))

    # 图例
    legend_x = w - int(base * 0.22)
    legend_y = font_size * 2
    legend_items = [
        (">=70%", (0, 180, 0)),
        ("30-70%", (255, 140, 0)),
        ("<30%", (220, 30, 30)),
    ]
    l_font_size = max(10, font_size * 3 // 4)
    try:
        l_font = ImageFont.truetype(FONT_PATH, l_font_size)
    except Exception:
        l_font = ImageFont.load_default()
    for text, clr in legend_items:
        draw.rectangle([legend_x, legend_y, legend_x + 16, legend_y + 10], fill=clr)
        draw.text((legend_x + 22, legend_y - 2), text, font=l_font, fill=clr)
        legend_y += l_font_size + 6

    # 保存
    out_path = os.path.join(result_dir, img_name)
    cv2.imwrite(out_path, cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
    return out_path


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "image")
    output_file = os.path.join(base_dir, "ocr_results.txt")
    result_dir = os.path.join(base_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    if not os.path.exists(image_dir):
        print(f"[错误] 图像目录不存在: {image_dir}")
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        print("[错误] 未找到图像文件")
        sys.exit(1)

    print(f"共找到 {len(image_files)} 张图像，初始化 EasyOCR 引擎...")

    reader = easyocr.Reader(["ch_sim", "en"], gpu=False)

    results_all = []
    total_start = time.time()

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("=" * 70 + "\n")
        f_out.write("图像文字检测结果报告\n")
        f_out.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_out.write(f"OCR 引擎: EasyOCR\n")
        f_out.write(f"支持语言: 简体中文 / English\n")
        f_out.write(f"图像总数: {len(image_files)}\n")
        f_out.write("=" * 70 + "\n\n")

        for idx, img_name in enumerate(image_files, 1):
            img_path = os.path.join(image_dir, img_name)
            print(f"[{idx}/{len(image_files)}] 正在处理: {img_name}")

            try:
                img_orig = cv2.imread(img_path)
                if img_orig is None:
                    continue
                h, w = img_orig.shape[:2]

                processed = preprocess_image(img_path)
                if processed is None:
                    continue

                start = time.time()
                detections = reader.readtext(processed, detail=1)
                elapsed = time.time() - start

                f_out.write(f"[{idx}] 文件: {img_name}\n")
                f_out.write(f"    图像尺寸: {w}x{h}\n")
                f_out.write(f"    处理耗时: {elapsed:.2f}s\n")

                if not detections:
                    f_out.write("    检测结果: 未检测到文字\n\n")
                else:
                    f_out.write(f"    检测到 {len(detections)} 处文字:\n\n")
                    for j, (bbox, text, confidence) in enumerate(detections, 1):
                        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                        conf_pct = confidence * 100

                        f_out.write(f"    [{j}] 位置: ({x1},{y1}) -> ({x2},{y2})\n")
                        f_out.write(f"        置信度: {conf_pct:.1f}%\n")
                        f_out.write(f"        文字内容: {text}\n\n")

                    all_texts = [text for _, text, _ in detections]
                    f_out.write(f"    --- 文本汇总 ---\n")
                    f_out.write(f"    {' | '.join(all_texts)}\n\n")

                    out_path = draw_ocr_boxes(img_orig, detections, img_name, result_dir)
                    print(f"    可视化 -> {out_path}")

                results_all.append({
                    "file": img_name,
                    "count": len(detections),
                    "time": elapsed
                })

            except Exception as e:
                f_out.write(f"[{idx}] 文件: {img_name}\n")
                f_out.write(f"    处理失败: {str(e)}\n\n")
                print(f"    [警告] 处理 {img_name} 时出错: {e}")
                results_all.append({"file": img_name, "count": 0, "time": 0})

        total_elapsed = time.time() - total_start
        total_texts = sum(r["count"] for r in results_all)
        f_out.write("=" * 70 + "\n")
        f_out.write("检测汇总\n")
        f_out.write("=" * 70 + "\n")
        f_out.write(f"总图像数: {len(image_files)}\n")
        f_out.write(f"检测到文字的总区域数: {total_texts}\n")
        f_out.write(f"总处理耗时: {total_elapsed:.2f}s\n")
        f_out.write(f"平均每张耗时: {total_elapsed / len(image_files):.2f}s\n")
        f_out.write("\n--- 各图像统计 ---\n")
        for r in results_all:
            f_out.write(f"  {r['file']}: {r['count']} 处文字, 耗时 {r['time']:.2f}s\n")

    print(f"\n检测完成! 共处理 {len(image_files)} 张图像，检测到 {total_texts} 处文字区域")
    print(f"TXT 结果: {output_file}")
    print(f"可视化结果: {result_dir}/")


if __name__ == "__main__":
    main()
