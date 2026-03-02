from pymycobot.mycobot280 import MyCobot280
import time
import image
import cv2
import YOLOV    
import mycode3
import numpy as np

mc = MyCobot280("COM5", 115200)  # 选COM

BIN_LOCATIONS = {   # 自定义放置点坐标库
    "bin1": [150, 160, 250, 180, 0, 0],
    "bin2": [0, 160, 250, 180, 0, 0],
    "bin3": [200, -100, 250, 150, -30, 0],
    "bin4": [150, -160, 250, 180, 0, 0],
    "bin5": [],
}

def go_to_observation():
    print("移动到观测位置，准备拍照...")
    mc.send_coords([112.7, -202.5, 263.2, -166.94, 3.13, -134.43], 50, 0) # 确保画面里没有机械臂   # [0, -20, -20, 0, 0, 0]
    time.sleep(5)
    print(mc.get_coords())

def grab_cube(u, v, bin_name):
    '''
    u, v: 物体坐标
    bin_name: 字符串格式，如 "bin1", "bin2"

    # --- 步骤1: 坐标转换 ---
    initial_arm_x = 150  # 像素中心(320)对应的物理X
    initial_arm_y = 0    # 像素中心(240)对应的物理Y
    initial_pixel_u = 320
    initial_pixel_v = 240
    zoom = 0.5  # 比例尺 (mm/px)
    u = initial_arm_x + (v - initial_pixel_v) * zoom  
    v = initial_arm_y + (u - initial_pixel_u) * zoom    
    '''
    
    safe_z = 270  # 移动时的安全高度
    move_z = 120   # 实际抓取（吸住）方块时的高度
    posture = [180, 0, 0] # 垂直向下
    print(f"--- 任务开始 ---")
    print(f"目标物体像素: ({u}, {v}) -> 物理坐标: ({u:.2f}, {v:.2f})")
    print(f"目标放置点: {bin_name}")

    # --- 步骤2：移动到物体正上方 ---
    print("1. 正在移动至物体上方...")
    mc.send_coords([u, v, safe_z] + posture, 50, 0)
    time.sleep(5)
    print(mc.get_coords())

    # --- 步骤3：下降并吸取 ---
    print("2. 执行下降吸附...")
    mc.send_coords([u, v, move_z] + posture, 20, 0)
    time.sleep(5) 
    mc.set_basic_output(5, 0) 

    # --- 步骤4：抬起物体 ---
    print("3. 提起物体...")
    mc.send_coords([u, v, safe_z] + posture, 50, 0)
    time.sleep(1)

    # --- 步骤5：移动到指定的 bin 并释放 ---
    if bin_name in BIN_LOCATIONS:
        bin_pos = BIN_LOCATIONS[bin_name]
        print(f"4. 移动至 {bin_name} 上方...")
        mc.send_coords(bin_pos, 50, 0) # 移动到 bin 的上方
        time.sleep(5)
        # print("coords: ", mc.get_coords())
        print(f"5. 释放物体...")
        mc.set_basic_output(5, 1) # 关闭吸泵
        time.sleep(1)
    else:
        print(f"【错误】: 找不到名为 {bin_name} 的放置点，请检查 BIN_LOCATIONS 字典！")
        mc.set_basic_output(5, 1)

    print("--- 任务结束 ---\n")



if __name__ == "__main__":
    go_to_observation()

    image.takepic()

    conf_thre = 0.3
      #置信度
    img = cv2.imread("./mytest.jpg")
    objects, path = mycode3.detect_objects(img, conf_thre = conf_thre)

    prompt = input("请输入指令：")
    # prompt = "I want the gun"
    text_embedding = mycode3.clip_text_encoder(prompt) #embed the text
    scores = []

    for obj in objects:
        crop = mycode3.crop_image(img, obj["bbox"])
        if crop is None:
            scores.append(-1)  #if naughty box, ignore it
            continue

        img_embedding = mycode3.clip_image_encoder(crop) #encode the bbox

        # cosine similarity (dot product because normalized)
        sim = (img_embedding @ text_embedding.T).item()
        scores.append(sim) #add cosine similarity to box

    if len(scores) == 0:
        print("❌ No detected objects for CLIP matching.")
        selected_bbox = None
    else:
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        SIM_THRESHOLD = 0.15  # reasonable default

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

    # object area: [to be defined]
    zone = [135, 0, 410, 270]  

    xcenter = (x1+x2)/2
    ycenter = (y1+y2)/2   # 中心像素
    x = 80+(ycenter-zone[1])/(zone[3]-zone[1])*150
    y = -80+(xcenter-zone[0])/(zone[2]-zone[0])*150   #坐标转换
    print(xcenter, ycenter, x, y)

    grab_cube(x, y, "bin1") # x, y