import os
import sys
import time
import easyocr
import cv2
import numpy as np


def preprocess_image(img_path):
    """图像预处理：缩放 + CLAHE增强 + 降噪，提高 OCR 识别准确率"""
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

    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened


def main():
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_results.txt")

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
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
