import os
import sys
import time
import easyocr
import cv2
import numpy as np


def preprocess_image(img_path):
    """图像预处理：缩放 + CLAHE增强 + 降噪"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    max_side = 2500
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened


def draw_ocr_boxes(img_orig, detections, img_name, result_dir):
    """在原图上绘制检测框和标注，保存到结果文件夹"""
    img_draw = img_orig.copy()
    h, w = img_draw.shape[:2]

    # 根据图像尺寸自适应调整参数
    base = max(h, w)
    box_thickness = max(2, base // 800)
    font_scale = base / 1200
    text_thickness = max(1, base // 1200)
    text_padding = base // 200
    max_text_len = 30  # 换行阈值

    # 已占用的标注区域，用于碰撞检测避免重叠
    occupied = []

    def is_overlap(y1_line, y2_line):
        margin = text_padding * 2
        for oy1, oy2 in occupied:
            if not (y2_line + margin < oy1 or y1_line - margin > oy2):
                return True
        return False

    for bbox, text, confidence in detections:
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        conf_pct = confidence * 100

        # 颜色选择：高置信度绿色，中等橙色，低置信度红色
        if conf_pct >= 70:
            color = (0, 200, 0)       # 绿
        elif conf_pct >= 30:
            color = (0, 140, 255)     # 橙
        else:
            color = (0, 0, 255)       # 红

        # 绘制检测框
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, box_thickness)

        # 构造标注文字
        label = f" {text} ({conf_pct:.1f}%)"

        # 计算标注位置：优先放在框上方，其次下方
        font_height = int(font_scale * 20)
        label_y_top = y1 - text_padding - font_height
        label_y_bot = y2 + text_padding

        # 尝试放上方
        if label_y_top > 0 and not is_overlap(label_y_top, y1):
            label_base_y = y1 - text_padding
            text_y = label_base_y - int(font_height * 0.3)
        else:
            # 放下方，如果贴图像底部则顶格
            label_base_y = y2 + text_padding + font_height
            if label_base_y > h:
                label_base_y = y2 - text_padding
                text_y = label_base_y - int(font_height * 0.3)
            else:
                text_y = label_base_y

        # 半透明背景框
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        bg_x1 = x1
        bg_x2 = x1 + tw + text_padding
        bg_y1 = text_y - th - text_padding // 2
        bg_y2 = text_y + text_padding // 2

        bg_x1 = max(0, bg_x1)
        bg_x2 = min(w, bg_x2)
        bg_y1 = max(0, bg_y1)
        bg_y2 = min(h, bg_y2)

        overlay = img_draw.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.addWeighted(overlay, 0.35, img_draw, 0.65, 0, img_draw)
        cv2.rectangle(img_draw, (bg_x1, bg_y1), (bg_x2, bg_y2), color, box_thickness)

        # 绘制文字
        cv2.putText(img_draw, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness,
                    cv2.LINE_AA)

        occupied.append((bg_y1, bg_y2))

    # 右上角图例
    legend_x = w - int(base * 0.22) - text_padding
    legend_y = text_padding * 3
    legend_items = [
        ("High (>=70%)", (0, 200, 0)),
        ("Medium (30-70%)", (0, 140, 255)),
        ("Low (<30%)", (0, 0, 255)),
    ]
    lw = int(base // 400) + 1
    lh = int(font_scale * 18)
    for label, color in legend_items:
        cv2.rectangle(img_draw, (legend_x, legend_y), (legend_x + 20, legend_y + 12), color, -1)
        cv2.putText(img_draw, label, (legend_x + 28, legend_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), max(1, lw - 1),
                    cv2.LINE_AA)
        legend_y += lh

    out_path = os.path.join(result_dir, img_name)
    cv2.imwrite(out_path, img_draw)
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

    print(f"共找到 {len(image_files)} 张图像，开始初始化 EasyOCR 引擎...")

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
        f_out.write(f"预处理: CLAHE增强 + 双边滤波降噪 + 锐化\n")
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
                    print(f"    可视化结果保存至: {out_path}")

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
