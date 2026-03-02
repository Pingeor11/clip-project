import cv2

def read_usb_capture():
    cap = cv2.VideoCapture(1) # 选择摄像头的编号：读取电脑摄像头，0；读取USB摄像头，1
    cv2.namedWindow('real_img', cv2.WINDOW_NORMAL) # 可以用鼠标拖动弹出的窗体
    while(cap.isOpened()):
        ret, frame = cap.read() # 读取摄像头的画面
        cv2.imshow('real_img', frame) # 真实图
        if cv2.waitKey(1) & 0xFF == ord('q'): # 按下'q'就退出
            break
    cap.release() # 释放画面
    cv2.destroyAllWindows()

def takepic():
    print("正在打开相机")

    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if (ret == 0):
        print("没有拍到照片！")
    
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print("width: ", width, "   height: ", height)

    cv2.imwrite("./mytest.jpg", frame)
    cap.release()

if __name__ == '__main__':
    takepic()

    # read_usb_capture()
