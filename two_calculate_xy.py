import math
import time
import numpy as np
import numpy as np
import openvr

from cam_pos_st import hmd_to_world, init_openvr, get_visual_controller_pos
from script_st import move_to_pos

"""
给iou做限制的
1.限制iou不要超过初始值
2.越小越好，尽量靠近
3.深度信息就是0，只保留xy，然后把这个转为世界坐标系，本质就是这么个东西
"""

def update_poses_and_get_hmd_matrix():
    """
    刷新一次 VRCompositor 的 pose，并返回最新的 HMD 矩阵（4×4 numpy）。
    """
    # 让 compositor 给你最新的所有 TrackedDevicePose_t
    poses = openvr.VRCompositor().waitGetPoses()[0]

    # HMD 在 openvr 的索引是 k_unTrackedDeviceIndex_Hmd = 0
    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
    if not hmd_pose.bPoseIsValid:
        return None

    # 把 TrackedDevicePose_t.mDeviceToAbsoluteTracking 转成 4×4 矩阵
    m = hmd_pose.mDeviceToAbsoluteTracking
    m = np.array([
        [m[0][0], m[0][1], m[0][2], m[0][3]],
        [m[1][0], m[1][1], m[1][2], m[1][3]],
        [m[2][0], m[2][1], m[2][2], m[2][3]],
        [0,       0,       0,       1     ]
    ], dtype=float)
    return m

def move_controller_in_depth_only(distance=0.2):
    """
    仅在深度方向（视线方向）上移动控制器，保持其在图像平面的相对位置不变，
    并避免上下抖动。

    Args:
        distance: 移动距离(米)，正值表示向前，负值表示向后

    Returns:
        tuple: 新的控制器位置(x,y,z) 或 None（如果失败）
    """
    init_openvr()
    # 1) 获取最新 HMD 世界矩阵
    hmd_matrix = hmd_to_world()
    if hmd_matrix is None:
        print("无法获取头盔位姿")
        return None

    # 2) 获取当前控制器位置
    controller_pos = get_visual_controller_pos()
    if controller_pos is None:
        print("无法获取控制器位置")
        return None

    # 3) 计算纯深度方向向量
    #    取 HMD 前向向量（Z 轴）
    fwd = hmd_matrix[:3, 2]
    #    去掉任何垂直分量，只保留在水平面和前后方向的成分
    up = hmd_matrix[:3, 1]
    fwd = fwd - np.dot(fwd, up) * up
    #    归一化
    norm = np.linalg.norm(fwd)
    if norm == 0:
        print("前向向量长度为0")
        return None
    fwd /= norm

    # 4) 计算新位置
    new_pos = controller_pos - fwd * distance

    # print(f"控制器当前位置: {controller_pos}")
    # print(f"深度移动向量: {fwd * distance}")
    #print(f"水平移动目标位置: {new_pos}")

    # 5) 发命令
    move_to_pos(new_pos[0],new_pos[1],new_pos[2])
    #print("深度移动成功" if success else "深度移动失败")
    return tuple(new_pos)

def move_controller_by_local_direction(x, y, z=0, distance=0.3):
    """
    根据头显本地坐标系中的 (x, y, z) 分量和给定距离，移动控制器到对应世界位置。

    参数:
        x: 本地 X 轴分量（正值→右，负值→左）
        y: 本地 Y 轴分量（正值→上，负值→下）
        z: 本地 Z 轴分量（正值→后，负值→前；在 OpenVR 中，Z+ 指向玩家）
        distance: 沿该方向移动的米数
    """
    init_openvr()
    hmd_m = hmd_to_world()
    if hmd_m is None:
        print("无法获取头盔位姿")
        return None

    pos = get_visual_controller_pos()
    if pos is None:
        print("无法获取控制器位置")
        return None
    pos = np.array(pos, dtype=float)

    # 构建本地方向向量（注意：OpenVR 本地 Z+ 朝玩家，故输入的 z 为正时后退，负时前进）
    local_vec = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(local_vec)
    if norm < 1e-6:
        print("输入方向向量过小")
        return None
    local_dir = local_vec / norm
    # 将本地方向转换到世界坐标系
    R = hmd_m[:3, :3]  # 取 3x3 旋转部分
    world_dir = R.dot(local_dir)

    # 归一化（避免数值累积误差）
    world_dir /= np.linalg.norm(world_dir)
    # 计算目标位置
    new_pos = pos + world_dir * distance
    move_to_pos(new_pos[0], new_pos[1], new_pos[2])
    return tuple(new_pos)



def calculate_distance_and_movement(obj1_box, obj2_box, tolerance=40):
    """
    计算两个物体之间的距离并判断是否需要移动

    Args:
        obj1_box: 第一个物体的边界框 (x, y, w, h)
        obj2_box: 第二个物体的边界框 (x, y, w, h)
        target_distance: 目标距离（像素）
        tolerance: 容许误差范围

    Returns:
        dict: 包含以下信息的字典:
            - distance: 两物体中心点距离
            - need_move: 是否需要移动
            - direction_x: x方向单位向量
            - direction_y: y方向单位向量
            - status: 状态描述
    """
    result = {
        "distance": None,
        "need_move": False,
        "direction_x": 0,
        "direction_y": 0,
        "status": "未检测到两个物体"
    }

    if obj1_box is None or obj2_box is None:
        return result

    # 计算中心点
    center1_x = obj1_box[0] + obj1_box[2] / 2
    center1_y = obj1_box[1] + obj1_box[3] / 2
    center2_x = obj2_box[0] + obj2_box[2] / 2
    center2_y = obj2_box[1] + obj2_box[3] / 2

    # 计算中心点之间的距离
    distance = np.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)

    # 计算方向向量 (从物体1到物体2)
    dx = center2_x - center1_x
    dy = center2_y - center1_y

    # 归一化方向向量（单位向量）
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    if magnitude > 0:
        direction_x = dx / magnitude
        direction_y = dy / magnitude
    else:
        direction_x = 0
        direction_y = 0

    # 判断是否需要移动
    need_move = distance > tolerance

    result = {
        "distance": distance,
        "need_move": need_move,
        "direction_x": direction_x,
        "direction_y": direction_y,
        "status": "需要拉近" if need_move else "距离合适"
    }

    return result

"""
提取一个基于头盔的前向向量，用来增加深度
"""
def get_hmd_forward_vector():
    """
    获取头盔的前向单位向量

    返回:
        forward_vector: 表示头盔朝向的单位向量 [x, y, z]
    """
    try:
        # 获取头盔的位姿矩阵
        init_openvr()
        hmd_pose = hmd_to_world()
        if hmd_pose is None:
            print("警告: 无法获取头盔位姿，使用默认前向向量")
            return [0, 0, -1]  # 默认使用-z方向作为前向

        # 从位姿矩阵中提取前向向量
        # 通常位姿矩阵的第三列表示前向方向
        # 注意：可能需要取负值，取决于坐标系的定义
        forward_vector = [
            -hmd_pose[0][2],  # 取负值是因为OpenVR中通常-Z是前向
            -hmd_pose[1][2],
            -hmd_pose[2][2]
        ]

        # 归一化向量
        magnitude = math.sqrt(sum(x * x for x in forward_vector))
        if magnitude > 0:
            forward_vector = [x / magnitude for x in forward_vector]
        print("前向向量是，",forward_vector)
        return forward_vector

    except Exception as e:
        print(f"提取头盔前向向量失败: {e}")
        return [0, 0, -1]  # 默认向量

# def move_controller_to_initial_position():
#     """
#     将控制器移动到头盔斜下方的可见位置，作为初始位置
#
#     返回:
#         bool: 移动是否成功
#     """
#     try:
#         # 初始化OpenVR
#         init_openvr()
#
#         # 获取头盔位置和朝向
#         hmd_matrix = hmd_to_world()
#         if hmd_matrix is None:
#             print("无法获取头盔位姿")
#             return False
#
#         # 提取头盔位置
#         hmd_pos = hmd_matrix[:3, 3]  # 位姿矩阵的第四列是位置
#
#         # 获取头盔的前向、右向和上向单位向量，前后老是不一样
#         raw_fwd = get_hmd_forward_vector()
#         forward_vector = [-v for v in raw_fwd]
#
#         right_vector = [hmd_matrix[0, 0], hmd_matrix[1, 0], hmd_matrix[2, 0]]
#         up_vector = [hmd_matrix[0, 1], hmd_matrix[1, 1], hmd_matrix[2, 1]]
#
#         # 计算控制器的目标位置：头盔前方、略下方、略右侧
#         # 距离: 前方0.5米
#         # 下方: 0.3米
#         # 右侧: 0.2米
#         forward_offset = 1
#         down_offset = 0
#         right_offset = 0
#
#         # 计算目标位置
#         target_pos = [
#             hmd_pos[0] + forward_vector[0] * forward_offset + right_vector[0] * right_offset - up_vector[
#                 0] * down_offset,
#             hmd_pos[1] + forward_vector[1] * forward_offset + right_vector[1] * right_offset - up_vector[
#                 1] * down_offset,
#             hmd_pos[2] + forward_vector[2] * forward_offset + right_vector[2] * right_offset - up_vector[
#                 2] * down_offset
#         ]
#
#         # 移动控制器到目标位置
#         move_to_pos(target_pos[0], target_pos[1], target_pos[2])
#         return True
#
#     except Exception as e:
#         print(f"移动控制器到初始位置失败: {e}")
#         return False

def move_controller_to_initial_offset(forward_offset=0.7, left_offset=0.2, down_offset=0.2):
    """
    将控制器移动到头盔前方 forward_offset 米、
    左侧 left_offset 米、下方 down_offset 米 的位置。

    参数:
        forward_offset: 正值表示沿视线向前
        left_offset:    正值表示沿本地 X-（左）
        down_offset:    正值表示沿本地 Y-（下）
    """
    init_openvr()
    hmd_m = hmd_to_world()
    if hmd_m is None:
        print("无法获取头盔位姿")
        return None

    # 头显世界坐标
    hmd_pos = hmd_m[:3, 3]

    # 提取本地轴
    right_vec   =  hmd_m[:3, 0]      # 本地 X+（右）
    up_vec      =  hmd_m[:3, 1]      # 本地 Y+（上）
    back_vec    =  hmd_m[:3, 2]      # 本地 Z+（朝玩家）
    forward_vec = -back_vec          # 视线方向（前方）

    # 计算合成位移向量
    offset_vec = (
        forward_vec   * forward_offset +
        (-right_vec)  * left_offset    +
        (-up_vec)     * down_offset
    )

    # 目标位置
    target_pos = hmd_pos + offset_vec

    # 调试输出
    print("HMD 世界位置:", hmd_pos.tolist())
    print("前向轴 vector:", forward_vec.tolist())
    print("右向轴 vector:", right_vec.tolist())
    print("上向轴 vector:", up_vec.tolist())
    print("合成 offset  :", offset_vec.tolist())
    print("目标位置    :", target_pos.tolist())
    move_to_pos(target_pos[0],target_pos[1],target_pos[2])

# if __name__ =="__main__":
#     init_openvr()
    #move_controller_by_local_direction(1,1)
    #move_controller_to_initial_offset()
#     move_controller_by_local_direction(1,1)
    #move_controller_in_depth_only(0.3)

