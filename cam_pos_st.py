import openvr
import numpy as np
import math
import cv2

#


# import script_st
# from openstero import pipeline_Openstero_real_depth


def init_openvr():
    """初始化OpenVR系统"""
    global vr_system
    openvr.init(openvr.VRApplication_Background)
    vr_system = openvr.VRSystem()

def shutdown_openvr():
    """关闭OpenVR系统"""
    openvr.shutdown()

"""
get_fov 获得左右眼的投影参数并且计算水平和垂直FOV，计算左眼的fov，默认为全局fov
calculate_focal_length：利用fov计算焦距
get_focal_length_from_steamvr：结合上面两种方法，得到焦距
"""
def get_fov():
    """从 SteamVR 获取左右眼的水平 & 垂直 FOV"""
    openvr.init(openvr.VRApplication_Scene)
    vr_system = openvr.VRSystem()

    # 获取左眼的投影参数 (left, right, bottom, top)
    left_proj_raw = vr_system.getProjectionRaw(openvr.Eye_Left)
    right_proj_raw = vr_system.getProjectionRaw(openvr.Eye_Right)

    # 计算 FOV（角度）
    def compute_fov(proj_raw):
        left, right, bottom, top = proj_raw
        fov_h = np.degrees(np.abs(np.arctan(right) - np.arctan(left)))  # 水平方向 FOV
        fov_v = np.degrees(np.abs(np.arctan(top) - np.arctan(bottom)))  # 垂直方向 FOV
        return fov_h, fov_v

    FOV_left = compute_fov(left_proj_raw)
    FOV_right = compute_fov(right_proj_raw)

    print(f"🎯 左眼 FOV: 水平 = {FOV_left[0]:.2f}°，垂直 = {FOV_left[1]:.2f}°")
    print(f"🎯 右眼 FOV: 水平 = {FOV_right[0]:.2f}°，垂直 = {FOV_right[1]:.2f}°")

    return {
        "FOV_left": FOV_left,
        "FOV_right": FOV_right
    }

# def calculate_focal_length(FOV_h, FOV_v, screen_width, screen_height):
#     """使用 FOV 计算 VR 头显的焦距 f_x 和 f_y（单位：像素）"""
#     # 将角度转换为弧度
#
#     FOV_h_rad = np.radians(FOV_h)
#     FOV_v_rad = np.radians(FOV_v)
#
#     # 计算焦距
#     f_x = screen_width / (2 * np.tan(FOV_h_rad / 2))
#     f_y = screen_height / (2 * np.tan(FOV_v_rad / 2))
#
#     print(f"🎯 计算得到的焦距: f_x = {f_x:.2f} px, f_y = {f_y:.2f} px")
#     return f_x, f_y

"""
获取静态和动态参数
get_oculus_static_params：获得瞳距，左右眼矩阵，屏幕分辨率
get_hmd_position_and_rotation：获得头显的四元数
# """

def get_projection():
    # 获取 OpenVR 的左眼和右眼投影矩阵
    left_projection = vr_system.getProjectionMatrix(openvr.Eye_Left, 0.1, 100.0)
    right_projection = vr_system.getProjectionMatrix(openvr.Eye_Right, 0.1, 100.0)

    # 转换为 NumPy 矩阵
    def convert_projection_matrix(projection):
        """ 将 OpenVR 返回的 projection 结构体转换成 NumPy 4x4 矩阵 """
        projection_np = np.array([
            [projection.m[0][0], projection.m[0][1], projection.m[0][2], projection.m[0][3]],
            [projection.m[1][0], projection.m[1][1], projection.m[1][2], projection.m[1][3]],
            [projection.m[2][0], projection.m[2][1], projection.m[2][2], projection.m[2][3]],
            [projection.m[3][0], projection.m[3][1], projection.m[3][2], projection.m[3][3]]
        ], dtype=np.float32)

        return projection_np

    # 应用转换
    left_projection_np = convert_projection_matrix(left_projection)
    right_projection_np = convert_projection_matrix(right_projection)

    return left_projection_np, right_projection_np


# def validate_depth(Z, min_depth=0.1, max_depth=10.0):
#     """验证深度值是否在合理范围内"""
#     if not (min_depth <= Z <= max_depth):
#         print(f"警告: 深度值 {Z}m 超出合理范围 [{min_depth}, {max_depth}]m")
#         return False
#     return True

def get_visual_controller_pos():
    # **获取手柄设备索引**
    init_openvr()
    controller_indexes = []
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vr_system.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
            controller_indexes.append(i)

    if not controller_indexes:
        print("❌ 未检测到 VR 手柄")
        return None

    # **获取所有设备的姿态（位置 + 旋转）**
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
    )

    for controller_id in controller_indexes:
        pose = poses[controller_id]
        if controller_id == 1:
            # **解析位置 (x, y, z)**
            matrix = pose.mDeviceToAbsoluteTracking
            x, y, z = matrix[0][3], matrix[1][3], matrix[2][3]
            return (x, y, z)

def calculate_world_position_2(
        left_point: np.ndarray,  # 左眼2D点坐标 [x, y]
        right_point: np.ndarray,  # 右眼2D点坐标 [x, y]
        left_projection: np.ndarray,  # 左眼投影矩阵 (4x4)
        right_projection: np.ndarray,  # 右眼投影矩阵 (4x4)
        left_eye_to_head: np.ndarray,  # 左眼到头部变换矩阵 (4x4)
        right_eye_to_head: np.ndarray,  # 右眼到头部变换矩阵 (4x4)
        hmd_to_world: np.ndarray  # HMD到世界空间变换矩阵 (4x4)
) -> np.ndarray:
    """
    使用视差直接计算VR中点的深度和世界坐标
    """
    # 1. 获取左右眼在头部空间中的位置
    left_eye_origin = np.array([0.0, 0.0, 0.0, 1.0])
    right_eye_origin = np.array([0.0, 0.0, 0.0, 1.0])

    left_eye_head = left_eye_to_head @ left_eye_origin
    right_eye_head = right_eye_to_head @ right_eye_origin

    # 归一化
    left_eye_head /= left_eye_head[3]
    right_eye_head /= right_eye_head[3]

    # 2. 计算两眼间距
    eye_separation = np.linalg.norm(right_eye_head[:3] - left_eye_head[:3])

    # 3. 将2D点转换为眼睛空间中的方向向量
    # 左眼
    left_ndc = np.array([left_point[0], left_point[1], -1.0, 1.0])
    left_eye_dir = np.linalg.inv(left_projection) @ left_ndc
    left_eye_dir /= left_eye_dir[3]
    left_eye_dir = left_eye_dir[:3] - np.array([0.0, 0.0, 0.0])  # 方向向量
    left_eye_dir /= np.linalg.norm(left_eye_dir)  # 单位化

    # 右眼
    right_ndc = np.array([right_point[0], right_point[1], -1.0, 1.0])
    right_eye_dir = np.linalg.inv(right_projection) @ right_ndc
    right_eye_dir /= right_eye_dir[3]
    right_eye_dir = right_eye_dir[:3] - np.array([0.0, 0.0, 0.0])  # 方向向量
    right_eye_dir /= np.linalg.norm(right_eye_dir)  # 单位化

    # 4. 将方向向量转换到头部空间
    # 创建方向点 (w=0 表示方向向量)
    left_dir_point = np.array([left_eye_dir[0], left_eye_dir[1], left_eye_dir[2], 0.0])
    right_dir_point = np.array([right_eye_dir[0], right_eye_dir[1], right_eye_dir[2], 0.0])

    # 变换方向向量到头部空间
    left_head_dir = left_eye_to_head @ left_dir_point
    right_head_dir = right_eye_to_head @ right_dir_point

    # 归一化方向向量
    left_head_dir = left_head_dir[:3]
    right_head_dir = right_head_dir[:3]
    left_head_dir /= np.linalg.norm(left_head_dir)
    right_head_dir /= np.linalg.norm(right_head_dir)
    # 5. 使用视差公式计算深度
    # 计算视线夹角（弧度）
    cos_angle = np.dot(left_head_dir, right_head_dir)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    # 视差公式: 深度 = 眼间距 / (2 * tan(视差角/2))
    # 视差角就是视线夹角
    depth = eye_separation / (2 * np.tan(angle / 2))
    # 6. 计算头部空间中的点位置
    # 使用左眼射线方向和计算出的深度
    head_point = left_eye_head[:3] + depth * left_head_dir
    # 7. 变换到世界空间
    # 构造齐次坐标
    head_point_homogeneous = np.array([head_point[0], head_point[1], head_point[2], 1.0])
    # 变换到世界空间
    world_point_homogeneous = hmd_to_world @ head_point_homogeneous
    # 归一化
    world_point_homogeneous /= world_point_homogeneous[3]
    # 取前三维
    world_point = world_point_homogeneous[:3]
    return world_point

# def calculate_world_position(
#         left_point: np.ndarray,  # 左眼2D点坐标 [x, y]
#         right_point: np.ndarray,  # 右眼2D点坐标 [x, y]
#         left_projection: np.ndarray,  # 左眼投影矩阵 (4x4)
#         right_projection: np.ndarray,  # 右眼投影矩阵 (4x4)
#         left_eye_to_head: np.ndarray,  # 左眼到头部变换矩阵 (4x4)
#         right_eye_to_head: np.ndarray,  # 右眼到头部变换矩阵 (4x4)
#         hmd_to_world: np.ndarray  # HMD到世界空间变换矩阵 (4x4)
# ) -> np.ndarray:
#     """
#     计算VR中某个点在世界空间中的3D坐标
#
#     参数:
#     - left_point: 左眼中看到的2D点坐标 [x, y]，范围[-1, 1]
#     - right_point: 右眼中看到的2D点坐标 [x, y]，范围[-1, 1]
#     - left_projection: 左眼投影矩阵
#     - right_projection: 右眼投影矩阵
#     - left_eye_to_head: 左眼到HMD变换矩阵
#     - right_eye_to_head: 右眼到HMD变换矩阵
#     - hmd_to_world: HMD到世界空间变换矩阵
#
#     返回:
#     - world_point: 点在世界空间中的3D坐标 [x, y, z]
#     """
#     print("-------calculate world postion-----------------------")
#     print("left_point",left_point)
#     print("right_point",right_point)
#     print("left_projection", left_projection)#列主序的left。是正确的
#     print("right_projection",right_projection)#列主序的right。是正确的
#     print("left_eye_to_head",left_eye_to_head)
#     print("right_eye_to_head",right_eye_to_head)
#     print("hmd_to_world",hmd_to_world)
#
#     def get_ray_from_eye_with_planes(point_2d, projection, eye_to_head, hmd_to_world,z_near=0.1,z_far=100,
#                                      ):
#         """
#         从 2D 归一化坐标点，利用近裁剪面(z_near)和远裁剪面(z_far)反投影为
#         世界坐标系下的一条射线 (ray_origin, ray_direction)
#         """
#
#         # 1. 构造 近平面 和 远平面 上的 4D 齐次坐标
#         point_near = np.array([point_2d[0], point_2d[1], z_near, 1.0])
#         point_far = np.array([point_2d[0], point_2d[1], z_far, 1.0])
#
#         # 2. 投影矩阵求逆，反投影到 眼睛空间
#         inv_proj = np.linalg.inv(projection)
#         eye_near = inv_proj @ point_near
#         eye_far = inv_proj @ point_far
#
#         # 齐次坐标归一化 (第四分量 w 不一定是1，需要除以 w)
#         eye_near /= eye_near[3]
#         eye_far /= eye_far[3]
#
#         # 3. 计算在眼睛空间的射线方向 (eye_ray_dir)
#         #    先求差，再只取前3维做归一化
#         eye_ray_dir = eye_far - eye_near
#         eye_ray_dir = eye_ray_dir[:3] / np.linalg.norm(eye_ray_dir[:3])
#
#         # 4. 将 near 点转换到头部坐标
#         head_near = eye_to_head @ eye_near
#         head_near /= head_near[3]
#
#         # 再转换到世界坐标
#         world_near = hmd_to_world @ head_near
#         world_near /= world_near[3]
#
#         # 5. 将 far 点 也转换到世界坐标
#         head_far = eye_to_head @ eye_far
#         head_far /= head_far[3]
#         world_far = hmd_to_world @ head_far
#         world_far /= world_far[3]
#
#         # 6. 用 world_far 和 world_near 的差 来确定射线方向
#         ray_origin = world_near[:3]  # 取齐次坐标前3维
#         ray_direction = world_far[:3] - world_near[:3]
#         ray_direction = ray_direction / np.linalg.norm(ray_direction)
#
#         return ray_origin, ray_direction
#
#     # 获取两条射线
#     left_origin, left_dir = get_ray_from_eye_with_planes(
#         left_point, left_projection, left_eye_to_head, hmd_to_world
#     )
#     right_origin, right_dir = get_ray_from_eye_with_planes(
#         right_point, right_projection, right_eye_to_head, hmd_to_world
#     )
#
#     # 计算两条射线的最近点
#     u = left_dir
#     v = right_dir
#     p0 = left_origin
#     p1 = right_origin
#
#     a = np.dot(u, u)
#     b = np.dot(u, v)
#     c = np.dot(v, v)
#     d = np.dot(u, p0 - p1)
#     e = np.dot(v, p0 - p1)
#
#     # 计算参数
#     s = (b * e - c * d) / (a * c - b * b)
#     t = (a * e - b * d) / (a * c - b * b)
#
#     # 计算最近点
#     p_left = p0 + s * u
#     p_right = p1 + t * v
#
#     # 取中点作为最终的3D位置
#     world_point = (p_left + p_right) / 2
#
#     return world_point

def eye_to_head():
    # 1获取 左眼 到 HMD 变换矩阵
    left_eye_to_head = vr_system.getEyeToHeadTransform(openvr.Eye_Left)

    # 2️ 获取 右眼 到 HMD 变换矩阵
    right_eye_to_head = vr_system.getEyeToHeadTransform(openvr.Eye_Right)


    # 3️ 转换为 NumPy 矩阵
    def convert_transform(mat):
        return np.array([
            [mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
            [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
            [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
            [0, 0, 0, 1]
        ])

    left_eye_to_head_matrix = convert_transform(left_eye_to_head)
    right_eye_to_head_matrix = convert_transform(right_eye_to_head)

    # 4️ 打印结果
    return left_eye_to_head_matrix,right_eye_to_head_matrix

def hmd_to_world():
    #  获取 HMD 的世界变换矩阵
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
    )

    #  HMD 位于索引 `openvr.k_unTrackedDeviceIndex_Hmd`
    pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]


    hmd_to_world_matrix = np.array([
        [pose.mDeviceToAbsoluteTracking[0][0], pose.mDeviceToAbsoluteTracking[0][1],
         pose.mDeviceToAbsoluteTracking[0][2], pose.mDeviceToAbsoluteTracking[0][3]],
        [pose.mDeviceToAbsoluteTracking[1][0], pose.mDeviceToAbsoluteTracking[1][1],
         pose.mDeviceToAbsoluteTracking[1][2], pose.mDeviceToAbsoluteTracking[1][3]],
        [pose.mDeviceToAbsoluteTracking[2][0], pose.mDeviceToAbsoluteTracking[2][1],
         pose.mDeviceToAbsoluteTracking[2][2], pose.mDeviceToAbsoluteTracking[2][3]],
        [0, 0, 0, 1]  # 齐次坐标
    ])
    # print("hmd_to_world function", hmd_to_world_matrix)
    return hmd_to_world_matrix

def calculate_3d_pos(cxl, cxr, cyl, cyr):
    # 这些矩阵需要从SteamVR API获取
    #投影矩阵
    left_projection, right_projection = get_projection()

    # 归一化2D点坐标
    left_point = np.array([cxl,cyl])
    right_point = np.array([cxr, cyr])

    #hmd to head and hmd to world
    left_eye_to_head_matrix, right_eye_to_head_matrix = eye_to_head()
    hmd_to_world_matrix = hmd_to_world()

    world_pos = calculate_world_position_2(#projection以下全部基于米
        left_point, right_point,
        left_projection, right_projection,
        left_eye_to_head_matrix, right_eye_to_head_matrix,
        hmd_to_world_matrix
    )

    print(f"世界坐标: {world_pos}")
    return world_pos


"""
基于openstereo来进行深度计算，left坐标即是视差图坐标
返回该点的实际深度
"""
def calculate_3d_pos_2(cxl, cxr, cyl, cyr):
    # 这些矩阵需要从SteamVR API获取
    #投影矩阵
    # left_projection, right_projection = get_projection()
    #
    # # 归一化2D点坐标
    # left_point = np.array([cxl,cyl])
    # right_point = np.array([cxr, cyr])

    # #hmd to head and hmd to world
    # left_eye_to_head_matrix, right_eye_to_head_matrix = eye_to_head()
    # hmd_to_world_matrix = hmd_to_world()
    #
    # world_pos = calculate_world_position_2(#projection以下全部基于米
    #     left_point, right_point,
    #     left_projection, right_projection,
    #     left_eye_to_head_matrix, right_eye_to_head_matrix,
    #     hmd_to_world_matrix
    # )
    focal_length, baseline, max_disp_default = get_vr_camera_parameters()

    real_depth = pipeline_Openstero_real_depth(baseline, focal_length, (cxl,cyl))

    print(f"实际深度: {real_depth}")
    return real_depth

def get_hmd_forward_vector():
    """
    使用 OpenVR 获取 HMD 的姿态，并返回一个相对于头盔前向的单位向量。
    参考向量 [0, 0, -1] 表示头盔默认的前方（注意：根据你的系统，前方可能有所不同）。
    """
    # 初始化 OpenVR
    openvr.init(openvr.VRApplication_Scene)
    vr_system = openvr.VRSystem()

    # 获取所有设备的姿态
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
    )

    hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
    if not hmd_pose.bPoseIsValid:
        print("HMD 姿态无效")
        openvr.shutdown()
        return None

    # 获取 HMD 的 3x4 姿态矩阵，并转换为 4x4 矩阵
    m = hmd_pose.mDeviceToAbsoluteTracking
    mat = np.array([
        [m[0][0], m[0][1], m[0][2], m[0][3]],
        [m[1][0], m[1][1], m[1][2], m[1][3]],
        [m[2][0], m[2][1], m[2][2], m[2][3]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 提取旋转部分（左上 3x3 矩阵）
    rot_matrix = mat[:3, :3]

    # 参考向量 [0, 0, -1]，表示头盔默认的正前方
    ref_forward = np.array([0, 0, -1])

    # 计算当前前向：旋转矩阵乘以参考向量
    forward = rot_matrix.dot(ref_forward)

    # 归一化得到单位向量
    norm = np.linalg.norm(forward)
    if norm != 0:
        forward_unit = forward / norm
    else:
        forward_unit = forward

    openvr.shutdown()
    return forward_unit* 0.1

#
def get_vr_camera_parameters(max_disp_default=192):
    """
    使用 OpenVR 获取 VR 环境下的相机参数：
      - focal_length：估计左眼投影矩阵 m[0][0]，作为焦距（单位：像素）
      - baseline：左右眼之间的距离（单位：米）
      - max_disp：最大视差值（通常取训练配置值，这里直接返回默认值）

    返回：
      (focal_length, baseline, max_disp)
    """
    # 初始化 OpenVR
    openvr.init(openvr.VRApplication_Scene)
    vr_system = openvr.VRSystem()

    # --- 获取焦距 ---
    # 设置近裁剪面和远裁剪面参数
    near = 0.1
    far = 1000.0
    # 获取左眼投影矩阵，返回一个 HmdMatrix44_t 对象
    left_proj = vr_system.getProjectionMatrix(openvr.Eye_Left, near, far)
    # 将其转换为 4x4 的 NumPy 数组
    left_proj_mat = np.array([
        [left_proj.m[0][0], left_proj.m[0][1], left_proj.m[0][2], left_proj.m[0][3]],
        [left_proj.m[1][0], left_proj.m[1][1], left_proj.m[1][2], left_proj.m[1][3]],
        [left_proj.m[2][0], left_proj.m[2][1], left_proj.m[2][2], left_proj.m[2][3]],
        [left_proj.m[3][0], left_proj.m[3][1], left_proj.m[3][2], left_proj.m[3][3]]
    ], dtype=np.float32)
    # 这里我们假设 left_proj_mat[0,0] 近似代表焦距（像素）
    focal_length = left_proj_mat[0, 0]

    # --- 获取基线（左右眼距离） ---
    # 获取左右眼到头部的变换矩阵
    left_transform = vr_system.getEyeToHeadTransform(openvr.Eye_Left)
    right_transform = vr_system.getEyeToHeadTransform(openvr.Eye_Right)

    def convert_openvr_matrix(mat):
        return np.array([
            [mat.m[0][0], mat.m[0][1], mat.m[0][2], mat.m[0][3]],
            [mat.m[1][0], mat.m[1][1], mat.m[1][2], mat.m[1][3]],
            [mat.m[2][0], mat.m[2][1], mat.m[2][2], mat.m[2][3]],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

    left_mat = convert_openvr_matrix(left_transform)
    right_mat = convert_openvr_matrix(right_transform)
    # 提取平移向量（眼睛位置），一般在矩阵的最后一列
    left_pos = left_mat[:3, 3]
    right_pos = right_mat[:3, 3]
    # 计算左右眼位置的欧氏距离，作为基线（单位：米）
    baseline = np.linalg.norm(right_pos - left_pos)

    # 关闭 OpenVR
    openvr.shutdown()

    return focal_length, baseline, max_disp_default


"""
Intrinsic_Parameters返回内参矩阵k，Stereo_rectification进行双目立体校正，先校正再做别的
"""
def Intrinsic_Parameters(horizontal_FOV, vertical_FOV, image_width, image_height):
    """
    Computes the intrinsic camera matrix (K) from the horizontal and vertical FOV and image size.
    """
    hFOV_rad = math.radians(horizontal_FOV)
    vFOV_rad = math.radians(vertical_FOV)
    # Compute focal lengths using: f = (w/2) / tan(horizontal_FOV/2)
    f_x = (image_width / 2.0) / math.tan(hFOV_rad / 2)
    f_y = (image_height / 2.0) / math.tan(vFOV_rad / 2)
    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([
        [f_x, 0,   cx],
        [0,   f_y, cy],
        [0,   0,   1]
    ], dtype=np.float32)
    return K

def Stereo_rectification(right_image,left_image):
    fov = get_fov()
    # 立体校正，内参和畸变系数，无畸变
    hFOV_left, vFOV_left = fov["FOV_left"]
    hFOV_right, vFOV_right = fov["FOV_right"]

    dist_left = np.zeros((5, 1), dtype=np.float32)
    dist_right = np.zeros((5, 1), dtype=np.float32)

    # 3. 根据图像尺寸计算内参矩阵
    image_height, image_width = left_image.shape[:2]
    image_size = (image_width, image_height)
    K_left = Intrinsic_Parameters(hFOV_left, vFOV_left, image_width, image_height)
    K_right = Intrinsic_Parameters(hFOV_right, vFOV_right, image_width, image_height)

    # 计算左右眼之间的相对变换矩阵：从右眼到左眼
    left_eye_to_head_matrix, right_eye_to_head_matrix = eye_to_head()
    relative_transform = np.linalg.inv(right_eye_to_head_matrix) @ left_eye_to_head_matrix

    # 提取旋转矩阵 R 和平移向量 T
    R = relative_transform[:3, :3]
    T = relative_transform[:3, 3]
    print(f"R={R},T={T}")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, dist_left,
        K_right, dist_right,
        image_size,
        R, T,
        alpha=0  # alpha=0 意味着裁剪图像以排除无效区域；可根据需要调整
    )

    # 生成校正映射矩阵
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        K_left, dist_left, R1, P1, image_size, cv2.CV_32FC1)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        K_right, dist_right, R2, P2, image_size, cv2.CV_32FC1)

    # 对左右图像进行重映射，得到校正后的图像
    rectified_left = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)


    # 显示校正后的图像以验证效果
    cv2.imshow("Rectified Left", rectified_left)
    cv2.imshow("Rectified Right", rectified_right)
    cv2.imwrite("OpenStereo/RectifiedLeft1.png", rectified_left)
    cv2.imwrite("OpenStereo/RectifiedRight1.png", rectified_right)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    return rectified_left,rectified_right


# if __name__ == "__main__":
# #
# #     # x,y,z = get_hmd_forward_vector()
# #     # print(x,y,z)
# #     # #script_st.move_to_pos(-0.3,0.4,-0.3)
# #     # script_st.move_to_pos(-0.1 , 0.05 , -0.5)
# #     # script_st.move_to_pos(-0.1 + x, 0.05 + y, -0.5 + z)
# #
#     script_st.move_to_pos(2.8,0,-2.8)
