from pymycobot.mycobot280 import MyCobot280
import time

mc = MyCobot280("COM5", 115200)  # 选COM


mc.send_coords([112.7, -202.5, 263.2, -166.94, 3.13, -134.43], 50, 0)
time.sleep(5)

mc.send_coords([209, 0, 200, 180, 0, 0], 50, 0)
time.sleep(5)

print(mc.get_coords())
