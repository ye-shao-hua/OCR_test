import os
import sys
import time
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/simhei.ttf"


def load_and_preprocess(img_path, max_side=1600, clip_limit=1.5):
    """加载图像并做自适应预处理：缩放 + 灰度 + CLAHE 增强"""
    img = cv2.imread(img_path)
    if img is None:
        return None, 1.0, 1.0, None

    h, w = img.shape[:2]
    scale_x, scale_y = 1.0, 1.0

    if max(h, w) > max_side:
        s = max_side / max(h, w)
        new_w, new_h = int(w * s), int(h * s)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scale_x = w / new_w
        scale_y = h / new_h
    else:
        img_resized = img

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced, scale_x, scale_y, img


def scale_detections(detections, sx, sy):
    """将检测框坐标从预处理图映射回原图"""
    scaled = []
    for bbox, text, conf in detections:
        pts = [[p[0] * sx, p[1] * sy] for p in bbox]
        scaled.append((pts, text, conf))
    return scaled


def iou(boxA, boxB):
    """两个边界框的 IoU（交并比），用于去重"""
    xa1 = max(min(p[0] for p in boxA), min(p[0] for p in boxB))
    ya1 = max(min(p[1] for p in boxA), min(p[1] for p in boxB))
    xa2 = min(max(p[0] for p in boxA), max(p[0] for p in boxB))
    ya2 = min(max(p[1] for p in boxA), max(p[1] for p in boxB))
    inter_w = max(0, xa2 - xa1)
    inter_h = max(0, ya2 - ya1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    areaA = (max(p[0] for p in boxA) - min(p[0] for p in boxA)) * (max(p[1] for p in boxA) - min(p[1] for p in boxA))
    areaB = (max(p[0] for p in boxB) - min(p[0] for p in boxB)) * (max(p[1] for p in boxB) - min(p[1] for p in boxB))
    return inter / min(areaA, areaB)


def merge_detections(pass1, pass2, iou_thresh=0.5):
    """合并两次检测结果，重叠框保留置信度更高的"""
    merged = list(pass1)
    for det_b in pass2:
        dup = False
        for i, det_a in enumerate(merged):
            if iou(det_a[0], det_b[0]) > iou_thresh:
                dup = True
                if det_b[2] > det_a[2]:
                    merged[i] = det_b
                break
        if not dup:
            merged.append(det_b)
    return merged


def dedup_detections(detections, iou_thresh=0.3):
    """移除高度重叠的重复检测，保留置信度更高的"""
    deduped = []
    for det in sorted(detections, key=lambda d: -d[2]):
        dup = False
        for existing in deduped:
            if iou(det[0], existing[0]) > iou_thresh:
                dup = True
                break
        if not dup:
            deduped.append(det)
    return deduped


def filter_low_quality(detections, conf_thresh=0.15, min_area_ratio=0.0001, img_w=1, img_h=1):
    """过滤低质量检测：置信度过低 或 框面积过小（噪声）"""
    img_area = img_w * img_h
    result = []
    for bbox, text, conf in detections:
        if conf < conf_thresh:
            continue
        w = max(p[0] for p in bbox) - min(p[0] for p in bbox)
        h = max(p[1] for p in bbox) - min(p[1] for p in bbox)
        if w * h < img_area * min_area_ratio:
            continue
        result.append((bbox, text, conf))
    return result


def detect_with_reader(reader, processed, sx, sy):
    """执行 EasyOCR 检测并映射回原图坐标"""
    raw = reader.readtext(processed, detail=1)
    return scale_detections(raw, sx, sy)


def draw_ocr_boxes(img_orig, detections, img_name, result_dir):
    """在原图上绘制检测框与标签，带回退字体支持"""
    h, w = img_orig.shape[:2]
    base = max(h, w)

    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    box_w = max(1, base // 1400)
    font_size = max(10, base // 65)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    def get_color(conf):
        if conf >= 70:
            return (80, 200, 80, 120)
        elif conf >= 30:
            return (240, 160, 60, 120)
        else:
            return (220, 80, 80, 120)

    occupied_regions = []

    for bbox, text, confidence in detections:
        x1 = int(min(p[0] for p in bbox))
        y1 = int(min(p[1] for p in bbox))
        x2 = int(max(p[0] for p in bbox))
        y2 = int(max(p[1] for p in bbox))
        conf_pct = confidence * 100
        color = get_color(conf_pct)

        draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=box_w)

        label = f"{text}({conf_pct:.0f}%)"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(2, font_size // 4)

        candidates = [
            (x1, y1 - th - pad * 2, x1 + tw + pad * 2, y1),
            (x1, y2, x1 + tw + pad * 2, y2 + th + pad * 2),
            (x2 + pad, y1, x2 + tw + pad * 3, y1 + th + pad * 2),
            (x1, y1, x1 + tw + pad * 2, y1 + th + pad * 2),
        ]

        best = None
        for bg_x1, bg_y1, bg_x2, bg_y2 in candidates:
            if bg_x2 > w or bg_y1 < 0 or bg_y2 > h:
                continue
            overlap = False
            for rx1, ry1, rx2, ry2 in occupied_regions:
                oxa = max(bg_x1, rx1)
                oya = max(bg_y1, ry1)
                oxb = min(bg_x2, rx2)
                oyb = min(bg_y2, ry2)
                if oxb > oxa and oyb > oya:
                    overlap = True
                    break
            if not overlap:
                best = (bg_x1, bg_y1, bg_x2, bg_y2)
                break

        if best is None:
            best = (x1, y1 - th - pad * 2, x1 + tw + pad * 2, y1)
        bg_x1, bg_y1, bg_x2, bg_y2 = best

        bg_x1 = max(0, bg_x1)
        bg_x2 = min(w, bg_x2)
        bg_y1 = max(0, bg_y1)
        bg_y2 = min(h, bg_y2)

        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color, outline=color[:3], width=1)
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(pil_img)

        tx = bg_x1 + pad
        ty = bg_y1 + (bg_y2 - bg_y1 - th) // 2
        draw.text((tx, ty), label, font=font, fill=(255, 255, 255))
        occupied_regions.append((bg_x1, bg_y1, bg_x2, bg_y2))

    # 图例
    lx = w - int(base * 0.15)
    ly = font_size // 2
    l_fs = max(9, font_size * 2 // 3)
    try:
        l_font = ImageFont.truetype(FONT_PATH, l_fs)
    except Exception:
        l_font = ImageFont.load_default()
    for label, clr in [(">=70", (80, 200, 80)), ("30-70", (240, 160, 60)), ("<30", (220, 80, 80))]:
        draw.rectangle([lx, ly, lx + 10, ly + 7], fill=clr)
        draw.text((lx + 14, ly - 2), label, font=l_font, fill=clr)
        ly += l_fs + 4

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

    ext_set = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in ext_set
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

                start = time.time()

                # 第一遍：标准预处理（CLAHE clipLimit=1.5，分辨率1600）
                processed1, sx1, sy1, _ = load_and_preprocess(img_path, max_side=1600, clip_limit=1.5)
                dets1 = detect_with_reader(reader, processed1, sx1, sy1) if processed1 is not None else []

                # 第二遍：更强对比度（CLAHE clipLimit=2.5，分辨率2000）——捕获第一遍漏掉的低对比度文字
                processed2, sx2, sy2, _ = load_and_preprocess(img_path, max_side=2000, clip_limit=2.5)
                dets2 = detect_with_reader(reader, processed2, sx2, sy2) if processed2 is not None else []

                # 合并两次检测 → 去重 → 过滤低质量
                detections = merge_detections(dets1, dets2, iou_thresh=0.45)
                detections = dedup_detections(detections, iou_thresh=0.3)
                detections = filter_low_quality(detections, conf_thresh=0.15, img_w=w, img_h=h)

                # 按位置排序（从上到下、从左到右）
                detections.sort(key=lambda d: (min(p[1] for p in d[0]), min(p[0] for p in d[0])))

                elapsed = time.time() - start

                f_out.write(f"[{idx}] 文件: {img_name}\n")
                f_out.write(f"    图像尺寸: {w}x{h}\n")
                f_out.write(f"    处理耗时: {elapsed:.2f}s\n")

                if not detections:
                    f_out.write("    检测结果: 未检测到文字\n\n")
                else:
                    f_out.write(f"    检测到 {len(detections)} 处文字:\n\n")
                    for j, (bbox, text, confidence) in enumerate(detections, 1):
                        x1 = int(min(p[0] for p in bbox))
                        y1 = int(min(p[1] for p in bbox))
                        x2 = int(max(p[0] for p in bbox))
                        y2 = int(max(p[1] for p in bbox))
                        conf_pct = confidence * 100

                        f_out.write(f"    [{j}] 位置: ({x1},{y1}) -> ({x2},{y2})\n")
                        f_out.write(f"        置信度: {conf_pct:.1f}%\n")
                        f_out.write(f"        文字内容: {text}\n\n")

                    all_texts = [t for _, t, _ in detections]
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
