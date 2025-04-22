import os
import queue
import threading
import time
from GroundingDINO import two_calculate_xy, indle
import difflib

import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

"""
1.第一步，抛弃双目测定，先移动到我得头盔视野前
2，第二步，测试他们中心点的距离，靠近+抑制到一定程度
3，第三步，利用midas进行深度测定，测定到ok为止(差不多就行了)
(结束)
"""
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)


# 全局队列和停止信号
detection_queue = queue.Queue(maxsize=1)  # 等待检测的帧队列
result_queue = queue.Queue(maxsize=2)  # 检测结果队列
stop_signal = threading.Event()  # 停止信号

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测设备

# 加载预处理器和模型
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def fuzzy_match(a: str, b: str, thresh: float = 0.6) -> bool:
    """
    字符串模糊匹配：当 a 和 b 的相似度 > thresh 时返回 True
    """
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > thresh


"""
zoom_crop_and_detect_single--放大检测单个目标
robust_dual_object_detection 检测两物体，未检测到再放大检测
"""
def zoom_crop_and_detect_single(frame, prev_box, object_name, scale_factor=3):
    """
    当目标尺寸太小时，通过裁剪放大后再次检测单个目标。
    tuple or None: 若成功返回映射回原图的边界框(x, y, w, h)，否则返回None。
    """
    height, width = frame.shape[:2]
    x, y, w, h = prev_box

    # 计算裁剪区域中心
    cx, cy = x + w // 2, y + h // 2

    # 计算新的裁剪区域尺寸
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)

    # 裁剪区域，防止超出边界
    x1 = max(cx - new_w // 2, 0)
    y1 = max(cy - new_h // 2, 0)
    x2 = min(cx + new_w // 2, width)
    y2 = min(cy + new_h // 2, height)

    # 裁剪并放大到原始帧的尺寸
    cropped_roi = frame[y1:y2, x1:x2]
    zoomed_roi = cv2.resize(cropped_roi, (width, height))

    # 在放大的图像上重新检测单个物体
    zoomed_rgb = cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2RGB)
    detection_result = Dino_detect_objects(zoomed_rgb, object_name, object_name)

    # 如果检测成功，则映射回原图坐标
    if detection_result["boxes"] is not None and len(detection_result["boxes"]) > 0:
        box = detection_result["boxes"][0]  # 取第一个框

        # 转换到像素坐标（放大后的尺寸）
        zx1, zy1, zx2, zy2 = box.tolist()
        zx1_px = int(zx1 * width)
        zy1_px = int(zy1 * height)
        zx2_px = int(zx2 * width)
        zy2_px = int(zy2 * height)

        # 缩放比例（映射回裁剪前的原图坐标）
        scale_x = (x2 - x1) / width
        scale_y = (y2 - y1) / height

        ox1_px = int(zx1_px * scale_x + x1)
        oy1_px = int(zy1_px * scale_y + y1)
        ox2_px = int(zx2_px * scale_x + x1)
        oy2_px = int(zy2_px * scale_y + y1)

        # 返回映射后的框
        mapped_box = (ox1_px, oy1_px, ox2_px - ox1_px, oy2_px - oy1_px)

        print("✅ 放大ROI后单物体检测成功。")
        return mapped_box
    else:
        print("⚠️ 放大ROI后单物体检测仍失败。")
        return None

def robust_dual_object_detection(frame, prev_obj1_box, prev_obj2_box, object1_name, object2_name, scale_factor=2):
    """
    稳健地检测两个物体，未检测到的则进行ROI放大再检测。用之前的框来代替
    """
    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 首先正常检测
    detection_result = Dino_detect_objects(frame_rgb, object1_name, object2_name)

    # 提取初次检测结果
    obj1_box, obj2_box, _ = draw_detection_results(
        frame, detection_result, width, height, object1_name, object2_name)

    # 物体1未检测到，启用ROI二次检测
    if obj1_box is None and prev_obj1_box is not None:
        obj1_box = zoom_crop_and_detect_single(frame, prev_obj1_box, object1_name, scale_factor)
        if obj1_box is None:
            print(f"[物体1 - {object1_name}] 二次检测仍失败，沿用上次框")
            obj1_box = prev_obj1_box  # 沿用上一帧位置

    # 物体2未检测到，启用ROI二次检测
    if obj2_box is None and prev_obj2_box is not None:
        obj2_box = zoom_crop_and_detect_single(frame, prev_obj2_box, object2_name, scale_factor)
        if obj2_box is None:
            print(f"[物体2 - {object2_name}] 二次检测仍失败，沿用上次框")
            obj2_box = prev_obj2_box  # 沿用上一帧位置

    return obj1_box, obj2_box


def new_robust_dual_object_detection(frame, prev_obj1_box, prev_obj2_box, object1_name, object2_name, scale_factor=2):
    """
    稳健地检测两个物体，未检测到的则进行ROI放大再检测
    处理位置漂移问题: 漂移过大时，不采纳检测结果，而是沿用上一帧位置或进行放大再检测
    """
    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 首先正常检测
    detection_result = Dino_detect_objects(frame_rgb, object1_name, object2_name)

    # 提取初次检测结果
    obj1_box, obj2_box, _ = draw_detection_results(
        frame, detection_result, width, height, object1_name, object2_name)

    # 物体1的处理
    if obj1_box is not None and prev_obj1_box is not None:
        # 检测是否存在极端漂移
        if not is_position_consistent(prev_obj1_box, obj1_box, alpha=1.5):
            print(f"[物体1 - {object1_name}] 检测到极端位置漂移，启动修复策略")

            # 在原位置附近区域尝试重新检测
            roi_obj1_box = zoom_crop_and_detect_single(frame, prev_obj1_box, object1_name, scale_factor)

            if roi_obj1_box is not None and is_position_consistent(prev_obj1_box, roi_obj1_box, alpha=1.8):
                # ROI再检测成功且位置合理，使用ROI检测结果
                print(f"[物体1 - {object1_name}] ROI重检测成功")
                obj1_box = roi_obj1_box
            else:
                # ROI再检测失败或仍不合理，使用上一帧位置
                print(f"[物体1 - {object1_name}] 采用上一帧位置")
                obj1_box = prev_obj1_box
    elif obj1_box is None and prev_obj1_box is not None:
        # 未检测到，尝试ROI二次检测
        obj1_box = zoom_crop_and_detect_single(frame, prev_obj1_box, object1_name, scale_factor)
        if obj1_box is None:
            print(f"[物体1 - {object1_name}] 二次检测仍失败，沿用上次框")
            obj1_box = prev_obj1_box  # 沿用上一帧位置

    # 物体2的处理 (与物体1逻辑相同)
    if obj2_box is not None and prev_obj2_box is not None:
        if not is_position_consistent(prev_obj2_box, obj2_box, alpha=1.5):
            print(f"[物体2 - {object2_name}] 检测到极端位置漂移，启动修复策略")

            roi_obj2_box = zoom_crop_and_detect_single(frame, prev_obj2_box, object2_name, scale_factor)

            if roi_obj2_box is not None and is_position_consistent(prev_obj2_box, roi_obj2_box, alpha=1.8):
                print(f"[物体2 - {object2_name}] ROI重检测成功")
                obj2_box = roi_obj2_box
            else:
                print(f"[物体2 - {object2_name}] 采用上一帧位置")
                obj2_box = prev_obj2_box
    elif obj2_box is None and prev_obj2_box is not None:
        obj2_box = zoom_crop_and_detect_single(frame, prev_obj2_box, object2_name, scale_factor)
        if obj2_box is None:
            print(f"[物体2 - {object2_name}] 二次检测仍失败，沿用上次框")
            obj2_box = prev_obj2_box  # 沿用上一帧位置

    return obj1_box, obj2_box

def update_center_distance(results):
    """
    从检测结果中计算前两个目标的中心点之间的欧氏距离，并更新结果字典。
    """
    if not results or not isinstance(results, list) or len(results) == 0:
        print("结果为空或格式错误")
        return results

    result = results[0]  # 取第一个结果字典
    boxes = result.get("boxes")
    if boxes is None:
        print("结果中没有找到 'boxes' 关键字")
        return results

    # 确保至少检测到两个目标
    if boxes.shape[0] < 2:
        print("检测到的目标不足两个，无法计算中心距离。")
        return results

    # 先将张量移动到CPU
    boxes_cpu = boxes.cpu()

    # 提取前两个目标的边界框
    box1 = boxes_cpu[0]  # [xmin, ymin, xmax, ymax]
    box2 = boxes_cpu[1]

    # 计算每个检测框的中心点
    center1_x = (box1[0].item() + box1[2].item()) / 2.0
    center1_y = (box1[1].item() + box1[3].item()) / 2.0

    center2_x = (box2[0].item() + box2[2].item()) / 2.0
    center2_y = (box2[1].item() + box2[3].item()) / 2.0

    #
    distance = np.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)
    print("----------------目前平面欧式欧式距离为:",distance,"-------------")

    # 将距离加入结果字典
    result["center_distance"] = distance
    return results

def Dino_detect_objects(frame, object1="white controller", object2="a TV"):
    """
    使用Grounding DINO检测两个目标物体
    Args:
        frame: 输入帧
        object1: 第一个物体的描述
        object2: 第二个物体的描述
    Returns:
        dict: 检测结果
    """
    # 组合两个目标的提示词
    text_prompt = f"Detect {object1} and the other is {object2}."

    # 对图片和文本进行预处理
    inputs = processor(images=frame, text=[text_prompt], return_tensors="pt",
                       padding=True, truncation=True).to(device)

    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 后处理，获取检测结果
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.35,  # 适当降低阈值以提高检出率
        text_threshold=0.25,
    )
    print("results",results,type(results))
    update_center_distance(results)
    return results[0]

def draw_detection_results(frame, result, width, height, object1_name, object2_name):
    """
    把这个
    """
    vis_frame   = frame.copy()
    obj1_box    = None
    obj2_box    = None
    obj1_score  = 0.0
    obj2_score  = 0.0

    # 只用一层循环！
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        x1, y1, x2, y2 = box.tolist()
        x = int(x1 * width)
        y = int(y1 * height)
        w = int((x2 - x1) * width)
        h = int((y2 - y1) * height)

        # 默认红框
        color = (0, 0, 255)

        # 1) 第一个物体：精确匹配
        if object1_name.lower() in label.lower() and score > obj1_score:
            obj1_score = score
            obj1_box   = (x, y, w, h)
            color      = (0, 255, 0)  # 绿框

        # 2) 第二个物体：子串或模糊匹配
        elif (object2_name.lower() in label.lower() or fuzzy_match(label, object2_name)) and score > obj2_score:
            obj2_score = score
            obj2_box   = (x, y, w, h)
            color      = (255, 0, 0)  # 蓝框

        # # 绘制框和标签
        # cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
        # cv2.putText(vis_frame, f"{label} ({score:.2f})", (x, y - 8),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return obj1_box, obj2_box, vis_frame

def detection_thread_function(object1_name, object2_name):
    """
    专门负责目标检测的线程
    Args:
        object1_name: 第一个物体名称
        object2_name: 第二个物体名称
    """
    print("检测线程启动")

    while not stop_signal.is_set():
        try:
            # 非阻塞获取最新帧，超时0.1秒
            frame_data = detection_queue.get(timeout=0.1)
            frame, frame_id = frame_data
            height, width, _ = frame.shape

            # 转RGB，Grounding DINO期望RGB输入
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 执行检测
            print(f"开始处理帧 {frame_id}")
            result = Dino_detect_objects(frame_rgb, object1_name, object2_name)

            # 将结果放入结果队列
            result_queue.put((result, frame_id, frame))
            print(f"帧 {frame_id} 处理完成")
        except queue.Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            print(f"检测线程错误: {e}")

    print("检测线程结束")


def convert_boxes_to_format(result, width, height, object1_name, object2_name):
    """
    从检测结果中提取指定物体的边界框
    Args:
        result: 检测结果
        width: 帧宽度
        height: 帧高度
        object1_name: 第一个物体名称
        object2_name: 第二个物体名称
    Returns:
        tuple: (物体1边界框, 物体2边界框)
    """
    obj1_box = None
    obj2_box = None
    obj1_score = 0
    obj2_score = 0

    if "boxes" in result and len(result["boxes"]) > 0:
        for i, (box, score, label) in enumerate(zip(result["boxes"], result["scores"], result["labels"])):
            if score < 0.35:
                continue

            # 归一化坐标转像素坐标
            x1, y1, x2, y2 = box.tolist()
            x = int(x1 * width)
            y = int(y1 * height)
            w = int((x2 - x1) * width)
            h = int((y2 - y1) * height)

            # 匹配物体名称
            if object1_name.lower() in label.lower() and score > obj1_score:
                obj1_box = (x, y, w, h)
                obj1_score = score

            if object2_name.lower() in label.lower() and score > obj2_score:
                obj2_box = (x, y, w, h)
                obj2_score = score

    return obj1_box, obj2_box


def process_tracking_with_sort(tracker, frame, detections, object_mapping, last_result, tracking_objects, object1_name,
                               object2_name, width, height):
    """
    使用SORT跟踪器处理当前帧，更新跟踪状态并返回跟踪结果

    Args:
        tracker: SORT跟踪器实例
        frame: 当前视频帧
        detections: 检测结果数组，格式为 [x1, y1, x2, y2, score]
        object_mapping: 检测索引到物体类型的映射字典
        last_result: 最近一次的检测结果
        tracking_objects: 当前正在跟踪的物体字典
        object1_name: 第一个物体的名称
        object2_name: 第二个物体的名称
        width: 帧宽度
        height: 帧高度

    Returns:
        tuple: (tracking_objects, obj1_box, obj2_box, display_frame)
            - tracking_objects: 更新后的跟踪对象字典
            - obj1_box: 物体1的边界框 (x, y, w, h) 或 None
            - obj2_box: 物体2的边界框 (x, y, w, h) 或 None
            - display_frame: 带有跟踪可视化的帧
    """
    # 更新SORT跟踪器
    track_bbs_ids = tracker.update(detections)

    # 从跟踪结果中提取物体1和物体2的边界框
    obj1_box = None
    obj2_box = None

    # 更新跟踪对象字典
    current_tracking_objects = {}

    # 处理跟踪结果
    for track in track_bbs_ids:
        track_id = int(track[4])  # 跟踪ID
        bbox = track[:4]  # [x1, y1, x2, y2]

        # 如果这是一个新ID，需要确定它对应哪个物体
        if track_id not in tracking_objects:
            # 找到与此跟踪框最匹配的检测框
            best_iou = 0
            best_type = None

            if last_result is not None and "boxes" in last_result and len(last_result["boxes"]) > 0:
                for i, (box, score, label) in enumerate(
                        zip(last_result["boxes"], last_result["scores"], last_result["labels"])):
                    if i in object_mapping:
                        # 转换检测框到像素坐标
                        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                        x1 = x1 * width
                        y1 = y1 * height
                        x2 = x2 * width
                        y2 = y2 * height
                        det_box = np.array([x1, y1, x2, y2])

                        # 计算IOU
                        iou = calculate_iou(bbox, det_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_type = object_mapping[i]

            # 如果找到匹配的物体类型
            if best_type:
                tracking_objects[track_id] = {"bbox": bbox, "type": best_type}
        else:
            # 如果ID已存在，只更新边界框
            tracking_objects[track_id] = {"bbox": bbox, "type": tracking_objects[track_id]["type"]}

        current_tracking_objects[track_id] = tracking_objects[track_id]

    tracking_objects = current_tracking_objects
    print(f"当前跟踪的物体数量: {len(tracking_objects)}")

    # 创建用于显示的帧
    display_frame = frame.copy()

    # 绘制跟踪结果并找出物体1和物体2的边界框
    for track_id, obj_info in tracking_objects.items():
        bbox = obj_info["bbox"]
        obj_type = obj_info["type"]

        # 从跟踪框提取坐标
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # 将跟踪框格式转换为(x, y, w, h)格式
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        # 保存对应物体的边界框
        if obj_type == "object1":
            obj1_box = (x, y, w, h)
            color = (0, 255, 0)  # 绿色
        elif obj_type == "object2":
            obj2_box = (x, y, w, h)
            color = (255, 0, 0)  # 蓝色
        else:
            color = (0, 0, 255)  # 红色

        # 绘制边界框
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

        # 添加标签
        label_text = f"{object1_name if obj_type == 'object1' else object2_name} (ID:{track_id})"
        cv2.putText(display_frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return tracking_objects, obj1_box, obj2_box, display_frame


import cv2

def draw_two_detections(frame,
                        box1=None, label1='',
                        box2=None, label2='',
                        color1=(0, 255, 0),   # 绿色
                        color2=(255, 0, 0),  # 蓝色
                        thickness=2):
    """
    在帧上绘制两个检测框，并返回绘制后的图像。
    """
    vis = frame.copy()

    for box, label, color in [(box1, label1, color1), (box2, label2, color2)]:
        if box is None:
            continue          # 跳过不存在的目标

        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        if label:
            cv2.putText(vis, label,
                        (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1,
                        lineType=cv2.LINE_AA)

    return vis

def main_3():
    # 导入SORT跟踪器
    from sort.sort import Sort

    #midas初始化
    indle.midas_setup()

    # 打开摄像头
    cap = cv2.VideoCapture(1)  # 使用摄像头1

    if not cap.isOpened():
        print("无法打开摄像头")
        return
    print("摄像头打开成功")

    two_calculate_xy.move_controller_to_initial_offset()#移动到初始位置
    print("移动到最初位置")

    # 获取目标物体名称
    object1_name = input("请输入第一个要跟踪的物体名称 (默认: gray game controller): ") or "gray game controller"
    object2_name = input("请输入第二个要跟踪的物体名称 (默认: VR screen): ") or "VR screen"
    frame_count = 0
    detection_interval = 90  # 每90帧检测一次
    # 跟踪相关变量
    tracking_objects = {}  # 存储跟踪ID和边界框
    #上一帧位置
    prev_obj1_box, prev_obj2_box = None, None

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break

        frame_count += 1
        display_frame = frame.copy()
        # 每隔detection_interval帧进行一次检测，或者没有跟踪对象时
        if frame_count % detection_interval == 1 or len(tracking_objects) == 0:
            print(f"----------------在第 {frame_count} 帧进行检测-------------------------------")

            # 转RGB，Grounding DINO期望RGB输入
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            obj1_box, obj2_box = new_robust_dual_object_detection(frame_rgb, prev_obj1_box, prev_obj2_box, object1_name, object2_name,scale_factor=2)
            print("_________obj1_box, obj2_box, ",obj1_box, obj2_box,"________________________")
            prev_obj1_box, prev_obj2_box = obj1_box, obj2_box # ←—— 更新“上一帧”的框，用当前检测（或补救后）的结果

            # 保存成图片
            display_frame = draw_two_detections(display_frame,
                        obj1_box, object1_name,obj2_box,object2_name)
            filename = f"image/frame_{frame_count:04d}.png"
            cv2.imwrite(filename, display_frame)
            #print(f"已保存：{filename}")

        if obj1_box is not None and obj2_box is not None:
            """深度调节"""
            depth_dict = indle.compare_depth(obj1_box, obj2_box, frame, depth_threshold=0.02)
            if depth_dict["need_move"]:
                print("--------------------开始深度移动一次------------------------")
                two_calculate_xy.move_controller_in_depth_only(1)
                for _ in range(10):
                    cap.grab()
            else:
                pass

            """抑制iou的移动"""
            iou_result = two_calculate_xy.calculate_distance_and_movement(obj1_box, obj2_box, tolerance=40)

            print("frame_count","need_move",iou_result["need_move"],iou_result["direction_x"],iou_result["direction_y"])
            if iou_result["need_move"]:
                print("----------开始平面移动------------------")
                #iou得上下和自然上下相反，所以取负
                two_calculate_xy.move_controller_by_local_direction(iou_result["direction_x"],-iou_result["direction_y"],0,0.1)
                # 清帧
                for _ in range(10):
                    cap.grab()
                continue
            else:
                pass
                # 清缓存

        cv2.imshow("多物体检测与跟踪", display_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC键退出
            print("用户中断")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

import math

def is_position_consistent(prev_box, new_box, alpha=1.2):
    """
    基于中心漂移判断位置一致性
    prev_box, new_box: (x, y, w, h)
    alpha: 最大漂移倍数（基于前一框对角线）
    """
    x0, y0, w0, h0 = prev_box
    x1, y1, w1, h1 = new_box

    # 计算中心
    cx0, cy0 = x0 + w0 / 2, y0 + h0 / 2
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2

    # 漂移距离
    dist = math.hypot(cx1 - cx0, cy1 - cy0)
    # 对角线长
    diag = math.hypot(w0, h0)
    return dist <= alpha * diag

def is_area_consistent(prev_box, new_box, area_range=(0.5, 2.0)):
    """
    基于面积比例判断大小一致性
    area_range: (min_ratio, max_ratio)
    """
    _, _, w0, h0 = prev_box
    _, _, w1, h1 = new_box
    area0 = w0 * h0
    area1 = w1 * h1
    if area0 == 0:
        return False
    ratio = area1 / area0
    return area_range[0] <= ratio <= area_range[1]



# 添加计算IOU的辅助函数
def calculate_iou(box1, box2):
    """
    计算两个边界框的IOU
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        float: IOU值
    """
    # 计算交集区域
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 交集区域为空的情况
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area

    # 计算IOU
    iou = intersection_area / union_area

    return iou

import math

"""
测试帧数是否合理的漂移
"""
def is_position_consistent(prev_box, new_box, alpha=2):
    """
    基于中心漂移判断位置一致性
    prev_box, new_box: (x, y, w, h)
    alpha: 最大漂移倍数（基于前一框对角线）
    """
    x0, y0, w0, h0 = prev_box
    x1, y1, w1, h1 = new_box

    # 计算中心
    cx0, cy0 = x0 + w0 / 2, y0 + h0 / 2
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2

    # 漂移距离
    dist = math.hypot(cx1 - cx0, cy1 - cy0)
    # 对角线长
    diag = math.hypot(w0, h0)
    return dist <= alpha * diag

def is_area_consistent(prev_box, new_box, area_range=(0.5, 2.0)):
    """
    基于面积比例判断大小一致性
    area_range: (min_ratio, max_ratio)
    """
    _, _, w0, h0 = prev_box
    _, _, w1, h1 = new_box
    area0 = w0 * h0
    area1 = w1 * h1
    if area0 == 0:
        return False
    ratio = area1 / area0
    return area_range[0] <= ratio <= area_range[1]



if __name__ == "__main__":
    main_3()
