import torch
from PIL import Image
import cv2
import numpy as np
import os
import sys
sys.path.insert(0, "./yolov5")   # 绝对路径，指向你下的仓库根目录
from yolov5.utils.general import non_max_suppression


# ============================================
# 主函数：输入numpy图像 → 输出检测结果和图片
# ============================================

def detect_objects(img_np, conf_thre, output_dir="results", save_name="detections.jpg"):
    # 原图尺寸
    h0, w0 = img_np.shape[:2]

    # BGR → RGB
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # ✅ 输入 YOLO：拉伸成 640×640
    img_resized = cv2.resize(img_rgb, (640, 640))

    img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(next(model.parameters()).device)

    # ✅ 模型推理
    pred = model(img_tensor)[0]

    # ✅ NMS
    pred = non_max_suppression(pred, conf_thres=conf_thre, iou_thres=0.45)[0]

    detected_objects = []
    best_dets = {}

    if pred is not None:
        # ✅ 缩放比例：从640×640→原图
        sx = w0 / 640.0
        sy = h0 / 640.0

        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            cls = int(cls.item())
            if cls in target_classes:
                class_name = target_class_names[target_classes.index(cls)]

                # ✅ 坐标缩回原图
                bx1 = x1 * sx
                by1 = y1 * sy
                bx2 = x2 * sx
                by2 = y2 * sy

                # ✅ 取置信度最高的
                if class_name not in best_dets or conf > best_dets[class_name][4]:
                    best_dets[class_name] = [bx1, by1, bx2, by2, float(conf), cls]

                detected_objects = [
                    {"class_name": c, 
                     "confidence": float(info[4]),
                     "bbox": [info[0], info[1], info[2], info[3]]}
                    for c, info in best_dets.items()
                ]

    # ✅ 画框：在原图上
    drawn = img_np.copy()
    for obj in detected_objects:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(drawn, f"{obj['class_name']} {obj['confidence']:.2f}",
                    (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, save_name)
    cv2.imwrite(output_path, drawn)

    print("="*50)
    if detected_objects:
        print(f"✅ 检测到 {len(detected_objects)} 个类别：")
        for i, obj in enumerate(detected_objects, 1):
            print(f"{i}. {obj['class_name']} ({obj['confidence']:.2f}) → {obj['bbox']}")
    else:
        print("❌ 未检测到目标类别")
    print(f"📁 结果已保存： {output_path}")
    print("="*50)

    return detected_objects, output_path

if __name__ == "__main__":

    # 设置检测所有类别 (ID 0-79)
    target_classes = list(range(80))  # 0到79的所有类别
    # COCO数据集的所有类别名称
    target_class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # ----------------------------------------
    # 加载 YOLOv5 模型（只需加载一次）
    # ----------------------------------------
    path = "./yolov5"
    model = torch.hub.load(path,
                        model='yolov5x',          # 与 pt 文件名保持一致
                        source='local',           # 关键
                        pretrained=False,          # 仍会加载同目录下的 yolov5x.pt
                        force_reload=False)       # 不需要再联网
    ckpt = torch.load("./yolov5x.pt", map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model'].state_dict())   # 官方权重格式
    model.eval()
    model.classes = target_classes

    conf_thre = 0.4  #置信度
    img = cv2.imread("./test.jpg")
    objects, path = detect_objects(img, conf_thre = conf_thre)