import mmap
import struct
import time
import numpy as np
import threading
from pynput.keyboard import Key, Listener, KeyCode



"""
自动移动，逼近目标坐标并保持 VR 设备活跃
"""

# 全局变量
x, y, z = 0, 0, 0
stop_event = threading.Event()  # 用于结束所有线程
listener_instance = None  # 存储 Listener 对象


def move_to_pos(x, y, z):
    global struct_format, shm, target_position, g_start
    struct_format = "fff???"
    shm = mmap.mmap(-1, struct.calcsize(struct_format), "DriverSharedMemory")

    # 初始化目标位置和启动标志
    target_position = (x, y, z)  # 目标坐标
    g_start = False  # 初始时不启动自动移动
    stop_event.clear()  # 清除终止事件

    # 创建线程
    mainloop_thread = threading.Thread(target=mainloop, daemon=True)
    listener_thread = threading.Thread(target=start_listener, daemon=True)

    # 启动线程
    mainloop_thread.start()
    listener_thread.start()

    # 示例：等待初始化后设置目标点并启动移动
    time.sleep(2)
    target_position = (x, y, z)  # 重新设置目标点（示例）
    g_start = True  # 启动自动移动

    # 等待线程结束（当 stop_event 触发后线程会自动退出）
    mainloop_thread.join()
    listener_thread.join()

    # 释放共享内存
    shm.close()


def is_position_close(data, target, threshold=1e-6):
    """ 判断 (x, y, z) 是否接近目标位置 """
    data_position = np.array(data[:3])
    target_position = np.array(target[:3])
    distance = np.linalg.norm(data_position - target_position)
    return distance < threshold


def sendsharememory(x, y, z, bclick, bdown, bup):
    """ 写入共享内存 """
    data = (x, y, z, bclick, bdown, bup)
    packed_data = struct.pack(struct_format, *data)
    shm.seek(0)
    shm.write(packed_data)


def mainloop():
    """ 逐步逼近目标点并持续写入共享内存 """
    print("🚀 开始自动移动")
    global x, y, z, g_start, target_position, listener_instance

    while not stop_event.is_set():
        if not g_start:
            time.sleep(0.01)
            continue  # 如果未启用，则暂停

        # 计算方向向量
        current_position = np.array([x, y, z])
        target_pos = np.array(target_position)
        direction = target_pos - current_position
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            print(f"🎯 已到达目标点: {target_position}")
            g_start = False  # 停止移动

            # 触发 stop_event，结束所有线程
            stop_event.set()
            if listener_instance is not None:
                listener_instance.stop()
            return -1

        # 计算步长
        step_size = 0.01  # 每次移动 0.01 米
        move_step = (direction / distance) * step_size

        # 更新坐标
        x += move_step[0]
        y += move_step[1]
        z += move_step[2]

        #print(f"🚀 逼近中: x={x:.4f}, y={y:.4f}, z={z:.4f}")

        # 写入共享内存
        sendsharememory(x, y, z, False, False, False)
        time.sleep(0.1)  # 控制移动速度


def on_press(key):
    """ 监听键盘输入 """
    global g_start
    if key == Key.enter:
        print("⏯️ 切换自动移动")
        g_start = not g_start


def start_listener():
    """ 启动键盘监听 """
    global listener_instance
    with Listener(on_press=on_press) as listener:
        listener_instance = listener
        listener.join()

