import torch
from PIL import Image
import cv2
import numpy as np
import os
import sys
sys.path.insert(0, "./yolov5")   # 绝对路径，指向你下的仓库根目录
from yolov5.utils.general import non_max_suppression
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F


# ============================================
# 主函数：输入numpy图像 → 输出检测结果和图片
# ============================================

def detect_objects(img_np, conf_thre, output_dir="results", save_name="detections.jpg"):
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

#LOAD CLIP
device = "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  #Loads CLIP Weights
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") #Loads preprocessing techniques
clip_model.eval() #Just using model, not trianing it

#ENCODE THE PROMPT
def clip_text_encoder(prompt: str):
    prompts = [
        prompt,
        f"a photo of {prompt}",
        f"something related to {prompt}",
        f"item that matches: {prompt}"
        f"close up photo of {prompt}"
    ]

    inputs = clip_processor(
        text=prompts,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)

    # normalize each, then average, then normalize again
    text_features = F.normalize(text_features, dim=-1)
    text_features = text_features.mean(dim=0, keepdim=True)
    text_features = F.normalize(text_features, dim=-1)

    return text_features


#Safety Crop so bbox isn't off the screen or overlapping
def crop_image(image_np, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image_np.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return image_np[y1:y2, x1:x2]

def clip_image_encoder(image_np):
    # BGR → RGB → PIL
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = clip_processor(
        images=pil_image,
        return_tensors="pt"
    ).to(device)  #[1,3,224,224], 3 for RGB

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    # Normalize for cosine similarity
    image_features = F.normalize(image_features, dim=-1)

    return image_features  # shape: [1, 512]




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
    img = cv2.imread("./gun.jpg")
    objects, path = detect_objects(img, conf_thre = conf_thre)

    prompt = "I want the gun"
    text_embedding = clip_text_encoder(prompt) #embed the text
    scores = []

    for obj in objects:
        crop = crop_image(img, obj["bbox"])
        if crop is None:
            scores.append(-1)
            continue

        img_embedding = clip_image_encoder(crop) #encode the bbox

        # cosine similarity (dot product because normalized)
        sim = (img_embedding @ text_embedding.T).item()
        scores.append(sim) #add cosine similarity to box

    if len(scores) == 0:
        print("❌ No detected objects for CLIP matching.")
        selected_bbox = None
    else:
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        SIM_THRESHOLD = 0.25  # reasonable default

        if best_score < SIM_THRESHOLD:
            print(f"❌ No object matches prompt (best score={best_score:.3f})")
            selected_bbox = None
        else:
            selected_bbox = objects[best_idx]["bbox"]
            print(f"✅ Selected object with CLIP score {best_score:.3f}")

    if selected_bbox is not None:
        x1, y1, x2, y2 = map(int, selected_bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, "CLIP SELECTED",
                    (x1, max(y1-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        cv2.imwrite("results/clip_selected.jpg", img)


        




