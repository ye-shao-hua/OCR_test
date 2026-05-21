# OCR_test

基于 Python + EasyOCR 实现的图像文字检测（OCR）项目。

## 功能

- 自动识别图像中的中文和英文文字内容
- 输出结构化 TXT 格式结果（含位置坐标、置信度、文字内容）
- 图像预处理增强：CLAHE 自适应直方图均衡化 + 双边滤波降噪 + 锐化

## 环境要求

- Python 3.9+
- EasyOCR 1.7+
- OpenCV 4.x
- PyTorch 2.x

## 安装依赖

```bash
pip install easyocr opencv-python numpy torch
```

## 使用方法

```bash
python main.py
```

将待检测图像放入 `image/` 目录，运行脚本后结果将输出至 `ocr_results.txt`。

## 目录结构

```
.
├── main.py           # OCR 检测主程序
├── image/            # 待检测图像目录（27 张）
├── ocr_results.txt   # 检测结果输出
├── 项目报告.md        # 项目报告（含算法分析）
└── README.md         # 本文件
```

## 检测效果

| 指标 | 数值 |
|------|------|
| 处理图像总数 | 27 |
| 检测到文字区域总数 | 45 |
| 总处理耗时（CPU） | 74.79s |

## 说明

- **检测引擎**：EasyOCR（基于 CRAFT + CRNN + CTC）
- **支持语言**：简体中文（ch_sim）、英文（en）
- 详细算法原理与实现分析见 `项目报告.md`
