from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("COM5", 115200)

# 危险操作确认：确保你已经手动把机械臂摆得非常正！
# 这一步会把当前的姿态定义为关节角度的 [0, 0, 0, 0, 0, 0]
mc.set_servo_calibration(1) # 校准 1 号电机
mc.set_servo_calibration(2) # 校准 2 号电机
mc.set_servo_calibration(3) # 校准 3 号电机
mc.set_servo_calibration(4) # 校准 4 号电机
mc.set_servo_calibration(5) # 校准 5 号电机
mc.set_servo_calibration(6) # 校准 6 号电机

print("所有关节零位已重置，请重启机械臂检查坐标是否正常。")