import os
import queue
import random
import threading
import time

from GroundingDINO import two_calculate_xy, indle
import difflib
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch.nn.functional as F
import cv2
import math

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

#中心点移动和向量特征传递
global move_history #上一帧中心点移动距离
move_history = 10000 #最初为极大值，方便初始化

global prev_obj1_feature
prev_obj1_feature = {
    "features": None,  # 特征向量张量
    "box": (0,0,0,0)
} # 边界框坐标
def fuzzy_match(a: str, b: str, thresh: float = 0.6) -> bool:
    """
    字符串模糊匹配：当 a 和 b 的相似度 > thresh 时返回 True
    """
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > thresh


"""
zoom_crop_and_detect_single--放大检测单个目标
robust_dual_object_detection 检测两物体，未检测到再放大检测
"""


def compare_feature_with_candidates(old_feature, new_feature_vectors, object_name=None, similarity_threshold=0.6):
    """
    比较单个旧特征向量与多个新特征向量的余弦相似度，找出最佳匹配吗，

    参数:
    old_feature: 上一帧的特征向量对象 (包含 "features" 和 "box" 等)
    new_feature_vectors: 当前帧的特征向量列表
    object_name: 可选，指定要比较的物体名称
    similarity_threshold: 可接受的最小相似度阈值

    返回:
    best_match: 字典，包含最佳匹配信息，如果没有足够相似的匹配则返回None
    """
    print("compare_feature_with_candidates")

    if not new_feature_vectors or old_feature is None or old_feature["features"] is None:
        return None

    # 过滤特定物体的特征向量（如果指定）
    filtered_vectors = new_feature_vectors
    if object_name is not None:
        filtered_vectors = [v for v in new_feature_vectors if object_name.lower() in v["label"].lower()]

    if not filtered_vectors:
        return None

    best_match_idx = -1
    best_similarity = -1.0
    best_combined_score = -1.0

    # 计算与每个新向量的余弦相似度
    for i, new_vec in enumerate(filtered_vectors):
        if new_vec["features"] is None:
            continue

        # 计算余弦相似度
        cos_sim = torch.mm(old_feature["features"], new_vec["features"].transpose(0, 1))
        similarity = (cos_sim.item() + 1) / 2  # 转换到0-1范围
        print("similarity",similarity)

        # 如果相似度低于阈值，考虑直接跳过这个候选
        if similarity < similarity_threshold:
            print(f"候选框 #{i} 特征相似度过低: {similarity:.3f} < {similarity_threshold}")
            continue

        # 添加空间一致性分数
        spatial_score = 0.5  # 默认中等空间得分
        if "box" in old_feature and "box" in new_vec:
            x1, y1, w1, h1 = old_feature["box"]
            x2, y2, w2, h2 = new_vec["box"]

            # 计算中心点
            cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
            cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2

            # 中心点距离
            dist = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
            max_dist = 300  # 假设最大合理距离
            spatial_score = max(0, 1 - (dist / max_dist))

            # 尺寸变化
            size_ratio = (w2 * h2) / (w1 * h1) if (w1 * h1) > 0 else float('inf')
            size_score = 0.0
            if 0.5 <= size_ratio <= 2.0:  # 允许尺寸变化一倍内
                # 归一化尺寸比例为0-1的分数，越接近1越好
                size_score = 1.0 - min(abs(1 - size_ratio), 1.0)

            # 综合分数：特征相似度65%，空间一致性25%，尺寸一致性10%
            combined_score = 0.65 * similarity + 0.25 * spatial_score + 0.10 * size_score

            print(f"候选框 #{i}: 特征相似度={similarity:.3f}, 空间分数={spatial_score:.3f}, "
                  f"尺寸分数={size_score:.3f}, 综合分数={combined_score:.3f}")
        else:
            combined_score = similarity
            print(f"候选框 #{i}: 特征相似度={similarity:.3f} (无空间信息)")

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_similarity = similarity
            best_match_idx = i

    # 只有在综合评分足够高的情况下才返回匹配
    if best_match_idx >= 0 and best_similarity >= similarity_threshold:
        best_match = {
            "match_idx": best_match_idx,
            "similarity": best_similarity,
            "combined_score": best_combined_score,
            "new_box": filtered_vectors[best_match_idx]["box"],
            "new_score": filtered_vectors[best_match_idx]["score"],
            "new_features": filtered_vectors[best_match_idx]["features"],
            "label": filtered_vectors[best_match_idx]["label"]
        }
        print(f"找到最佳匹配: 特征相似度={best_similarity:.3f}, 综合分数={best_combined_score:.3f}")
        return best_match

    print("未找到足够相似的匹配")
    return None

"""附属函数，用来检测的"""


def visualize_detection_results(frame, best_match_box, output_path=None, scale=2):
    """
    可视化检测结果，只显示最佳匹配框，并保存结果图像

    Args:
        frame: 原始尺寸的图像帧
        best_match_box: 基于放大图像的边界框坐标 (x, y, w, h)
        output_path: 输出图像路径，如果为None则生成自动的文件名
        scale: 图像缩放比例，与best_match_box坐标的缩放比例相同

    Returns:
        保存图像的路径
    """
    import os
    import time
    import cv2

    # 创建可视化图像副本
    vis_frame = frame.copy()

    # 如果有边界框，需要将坐标从放大尺寸转换回原始尺寸
    if best_match_box is not None:
        x, y, w, h = best_match_box

        # 将边界框坐标从放大尺寸转换回原始尺寸
        x_original = int(x / scale)
        y_original = int(y / scale)
        w_original = int(w / scale)
        h_original = int(h / scale)

        # 在原始尺寸图像上绘制边界框
        cv2.rectangle(vis_frame,
                      (x_original, y_original),
                      (x_original + w_original, y_original + h_original),
                      (0, 255, 0), 2)

        # 标记为"Best Match"
        cv2.putText(vis_frame, "Best Match",
                    (x_original, y_original - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # 原始尺寸的文本大小
                    (0, 255, 0), 1)

        status_text = "Match_Found"
    else:
        # 如果没有最佳匹配，显示提示信息
        cv2.putText(vis_frame, "No Match Found",
                    (10, 30),  # 原始尺寸的位置
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255), 2)

        status_text = "No_Match"

    # 然后调整整个图像大小用于保存/显示（此时已包含绘制的边界框）
    display_height, display_width = vis_frame.shape[:2]
    display_frame = cv2.resize(vis_frame, (display_width * scale, display_height * scale))

    # 添加时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # # 生成输出路径，如果未指定
    # if output_path is None:
    #     # 确保images目录存在
    #     os.makedirs("images", exist_ok=True)
    #     random_number = random.randint(0, 9999)
    #     output_path = f"images/detection_{status_text}_{timestamp}_{random_number}.jpg"
    #
    # # 保存图像
    # success = cv2.imwrite(output_path, display_frame)
    #
    # if success:
    #     print(f"检测结果已保存至: {output_path}")
    # else:
    #     print(f"警告: 无法保存图像到 {output_path}")

    return output_path
def full_image_detect_single(frame, object_name, scale_factor=2):
    """
    将整个图像放大，然后在放大后的图像上检测单个目标。
    使用特征向量匹配提高检测稳定性。
    """
    # 获取原始图像尺寸
    global prev_obj1_feature
    height, width = frame.shape[:2]

    # 计算放大后的尺寸
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # 将整个图像放大
    upscaled_frame = cv2.resize(frame, (new_width, new_height))

    # 将放大的图像转换为RGB（DINO需要RGB输入）
    upscaled_rgb = cv2.cvtColor(upscaled_frame, cv2.COLOR_BGR2RGB)

    # 在放大后的图像上进行目标检测（只检测指定的单个目标）
    #print("upscaled_rgb, object_name, object_name",upscaled_rgb, object_name, object_name)
    detection_results, feature_vectors = Dino_detect_objects_1(upscaled_rgb, object_name, object_name,0.1,0.1)

    print(f"检测到 {len(feature_vectors)} 个候选框")

    # 将特征向量的坐标映射回原始尺寸
    for vec in feature_vectors:
        print("label1111111，显示具体召唤了什么",vec["box"],vec["label"])
        if vec["box"] is not None:
            x, y, w, h = vec["box"]
            visualize_detection_results(frame,vec["box"])
            # 映射回原始尺寸
            vec["box"] = (int(x / scale_factor),int(y / scale_factor),int(w / scale_factor),int(h / scale_factor))

    # 如果有上一帧特征，使用特征匹配
    # 有初始值
    if prev_obj1_feature['box'] != (0, 0, 0, 0) and prev_obj1_feature['features'] is not None:
        # 使用改进的特征匹配函数
        best_match = compare_feature_with_candidates(prev_obj1_feature, feature_vectors, object_name,
                                                     similarity_threshold=0.6)

        if best_match is not None:
            detected_box = best_match["new_box"]
            best_feature = {
                "features": best_match["new_features"],
                "box": best_match["new_box"]
            }
            print(f"✅ 使用特征匹配成功找到 {object_name}，特征相似度: {best_match['similarity']:.2f}")
            return detected_box, best_feature
    else:
        if feature_vectors:
            # 过滤出与目标名称匹配的特征向量
            filtered_vectors = [v for v in feature_vectors if object_name.lower() in v["label"].lower()]

            if filtered_vectors:
                # 选择置信度最高的候选框
                best_candidate = max(filtered_vectors, key=lambda x: x["score"])
                detected_box = best_candidate["box"]
                best_feature = {
                    "features": best_candidate["features"],
                    "box": best_candidate["box"]
                }
                print(f"✅ 首次检测，使用置信度最高的框: {best_candidate['score']:.2f}")
                prev_obj1_feature = best_feature
                return detected_box, best_feature

    # 如果没有匹配结果，返回None
    print(f"⚠️ 全图放大检测未能找到匹配的 {object_name}。")
    return None, None


def extract_region_features_dino(region):
    """
    从区域中提取特征向量，适用于GroundingDINO模型

    Args:
        region: BGR格式的图像区域

    Returns:
        tensor: 特征向量
    """
    # 转换为RGB
    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

    # 使用处理器处理图像，添加必要的文本输入
    dummy_text = "object"  # 一个简单的占位文本
    inputs = processor(images=region_rgb, text=[dummy_text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # 尝试直接使用模型进行前向传播
        outputs = model(**inputs)  # 不要添加output_hidden_states参数

        # 提取视觉特征 - 尝试几种可能的方法

        # 如果模型输出有last_hidden_state
        if hasattr(outputs, 'last_hidden_state'):
            # 取[CLS]向量或第一个令牌
            features = outputs.last_hidden_state[:, 0, :]

        # 特征归一化
        features = F.normalize(features, p=2, dim=1).cpu()

        return features


def new_robust_dual_object_detection(frame, prev_obj1_box, prev_obj2_box, object1_name, object2_name,scale_factor=2):
    """
    稳健地检测两个物体：
    1. 专注于物体1的检测质量，使用全图放大方法进行重检测
    2. 物体2直接沿用上一帧结果
    move_history:上一次的物体一的移动距离
    """
    print("new_robust_dual_object_detection")
    global move_history,prev_obj1_feature # 声明这个变量是全局的
    # print(f"新一帧的move_history={move_history}")
    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected = True

    # 首先正常检测
    detection_result,feature_vectors = Dino_detect_objects_0(frame_rgb, object1_name, object2_name)
    # 提取初次检测结果
    obj1_box, obj2_box, _ = draw_detection_results(
        frame, detection_result, width, height, object1_name, object2_name)
    print(f"初次检测，检测obj1_box, obj2_box,prev_obj1_box：{obj1_box},{obj2_box},{prev_obj1_box}")
    # 物体2的处理 - 简单沿用上一帧位置
    if prev_obj2_box is not None:
        print(f"[物体2 - {object2_name}] 使用上一帧位置")
        obj2_box = prev_obj2_box

    # 物体1的处理
    print("处理第一个物体")
    if obj1_box is None and prev_obj1_box is None:#前一帧没有，这一帧也没有检测到，最初情况
        return None,obj2_box, True
    elif prev_obj1_box is not None:
        if obj1_box is not None :#前一帧数有，这一帧数有，检测 is_area_consistent
            is_area_consistent, current_dist = is_position_consistent(prev_obj1_box, obj1_box, move_history,2)
            print(f"is_area_consistent:{is_area_consistent},move_history:{move_history}","current_dist",current_dist)
            if is_area_consistent : #如果一致
                if current_dist >= 10:#防止两帧返回同一张图
                    move_history = current_dist
                return obj1_box,obj2_box,True

        #检测不可靠，放大重检测,包括没有检测到当前物体，和检测到但是不一致
        print(f"[物体1 - {object1_name}] 检测不可靠，使用全图放大重检测")
        # 全图放大重检测,此时已经经过了特征向量比较
        upscaled_obj1_box, best_feature = full_image_detect_single(frame, object1_name)
        print(f"此时已经经过了特征向量比较,upscaled_obj1_box：{upscaled_obj1_box},prev_obj1_box:{prev_obj1_box}")
        if upscaled_obj1_box is None: # 特征向量可以了，然后呢
            return obj1_box,obj2_box,True

        # 如果重检测成功且位置合理，使用重检测结果
        is_consistent, new_move_history =is_position_consistent(prev_obj1_box, upscaled_obj1_box, move_history,2)
        print(f"重检测条件 is_consistent:{is_consistent},move_history:{move_history}")

        if upscaled_obj1_box is not None and is_consistent:
            move_history = new_move_history
            obj1_box = upscaled_obj1_box
            prev_obj1_feature = best_feature#更改全局变量
            print(f"[物体1 - {object1_name}] 全图重检测成功,重新画图，更新全局变量prev_obj1_feature")

        else:
            # 重检测失败或位置不合理，沿用上一帧位置
            if prev_obj1_box is not None:
                print(f"[物体1 - {object1_name}] 重检测失败，沿用上一帧位置")
                obj1_box = prev_obj1_box
            else:
                print(f"[物体1 - {object1_name}] 首次检测完全失败")
                detected = False

    elif prev_obj1_box is None:#有当前帧，没有过去帧，保存当前的当量
        if prev_obj1_feature['box'] == (0, 0, 0, 0) and prev_obj1_feature['features'] is None:  # 初始化这个prev_obj1_feature\
            for item in feature_vectors:
                if item["box"] == obj1_box:
                    new_features = item["features"]
                    break
            prev_obj1_feature = {
                "features": new_features,  # 特征向量张量
                "box": obj1_box
            }


    return obj1_box, obj2_box, detected


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
    print("目前平面欧式欧式距离为:",distance)

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
    #print("results",results,type(results))
    update_center_distance(results)
    return results[0]


def Dino_detect_objects_0(frame, object1="white controller", object2="a TV", box_threshold=0.35,text_threshold=0.25):
    """
    使用Grounding DINO检测两个目标物体
    Args:
        frame: 输入帧
        object1: 第一个物体的描述
        object2: 第二个物体的描述
    Returns:
        tuple: (检测结果, 特征向量数组),返回所有obj1的特征向量
    """
    # 组合两个目标的提示词
    text_prompt = f"Detect {object1} and the other is {object2}."

    # 对图片和文本进行预处理
    inputs = processor(images=frame, text=[text_prompt], return_tensors="pt",
                       padding=True, truncation=True).to(device)

    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)

        # 提取视觉特征
        if hasattr(model, 'vision_model'):
            # 从视觉编码器提取特征
            visual_features = model.vision_model(inputs.pixel_values).pooler_output

    # 后处理，获取检测结果
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,  # 适当降低阈值以提高检出率
        text_threshold=text_threshold,
    )
    # 更新中心距离
    update_center_distance(results)

    # 提取单个框的特征向量数组
    feature_vectors = []

    if results[0]["boxes"] is not None and len(results[0]["boxes"]) > 0:
        height, width = frame.shape[:2]

        for i, (box, score, label) in enumerate(zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"])):
            # 转换坐标
            x1, y1, x2, y2 = box.tolist()
            x = int(x1 * width)
            y = int(y1 * height)
            w = int((x2 - x1) * width)
            h = int((y2 - y1) * height)

            # 确保区域有效
            if w > 10 and h > 10:
                # 提取该区域
                region = frame[y:y + h, x:x + w]
                features = extract_region_features_dino(region)
                # 添加到特征向量列表
                feature_vectors.append({
                    "box": (x, y, w, h),
                    "score": float(score),
                    "label": label,
                    "features": features
                })
    return results[0], feature_vectors

def Dino_detect_objects_1(frame, object1="white controller", object2="a TV", box_threshold=0.35,text_threshold=0.25):
    """
    使用Grounding DINO检测两个目标物体
    Args:
        frame: 输入帧
        object1: 第一个物体的描述
        object2: 第二个物体的描述
    Returns:
        tuple: (检测结果, 特征向量数组),返回所有obj1的特征向量,与0的区别是返回所有结果
    """
    # 组合两个目标的提示词
    text_prompt = f"Detect {object1} and the other is {object2}."

    # 对图片和文本进行预处理
    inputs = processor(images=frame, text=[text_prompt], return_tensors="pt",
                       padding=True, truncation=True).to(device)

    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)

        # 提取视觉特征
        if hasattr(model, 'vision_model'):
            visual_features = model.vision_model(inputs.pixel_values).pooler_output

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    update_center_distance(results)

    # 提取单个框的特征向量数组
    feature_vectors = []
    print(len(results))
    for result in results :
        if results[0]["boxes"] is not None and len(results[0]["boxes"]) > 0:
            height, width = frame.shape[:2]

            for i, (box, score, label) in enumerate(zip(result["boxes"], result["scores"], result["labels"])):
                # 转换坐标
                x1, y1, x2, y2 = box.tolist()
                x = int(x1 * width)
                y = int(y1 * height)
                w = int((x2 - x1) * width)
                h = int((y2 - y1) * height)

                # 确保区域有效
                if w > 10 and h > 10:
                    # 提取该区域
                    region = frame[y:y + h, x:x + w]
                    features = extract_region_features_dino(region)
                    #print("在feature_vectors里取得的结果",(x, y, w, h))
                    # 添加到特征向量列表
                    feature_vectors.append({
                        "box": (x, y, w, h),
                        "score": float(score),
                        "label": label,
                        "features": features
                    })
    return results[0], feature_vectors


def draw_detection_results(frame, result, width, height, object1_name, object2_name):
    """
    把这个
    """
    vis_frame = frame.copy()
    obj1_box = None
    obj2_box = None
    obj1_score = 0.0
    obj2_score = 0.0

    # 只用一层循环！
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        x1, y1, x2, y2 = box.tolist()
        x = int(x1 * width)
        y = int(y1 * height)
        w = int((x2 - x1) * width)
        h = int((y2 - y1) * height)

        # 1) 第一个物体：精确匹配
        if (object1_name.lower() in label.lower() or fuzzy_match(label, object1_name)) and score > obj1_score:
            obj1_score = score
            obj1_box = (x, y, w, h)

        # 2) 第二个物体：子串或模糊匹配
        elif (object2_name.lower() in label.lower() or fuzzy_match(label, object2_name)) and score > obj2_score:
            obj2_score = score
            obj2_box = (x, y, w, h)
    return obj1_box, obj2_box, vis_frame

def draw_two_detections(frame,box1=None, label1='',box2=None, label2='',color1=(0, 255, 0),  color2=(255, 0, 0),thickness=2):
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
            cv2.putText(vis, label,(x, y - 6),cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1,lineType=cv2.LINE_AA)
    return vis

def main_3():
    #midas初始化
    indle.midas_setup()

    # 打开摄像头
    two_calculate_xy.move_controller_to_initial_offset()#移动到初始位置
    print("移动到最初位置")
    while True:
        # 获取目标物体名称
        object1_name = input("请输入第一个要跟踪的物体名称 (默认: gray game controller): ") or "gray game controller"
        object2_name = input("请输入第二个要跟踪的物体名称 (默认: VR screen): ") or "VR screen"
        frame_count = 0
        detection_interval = 90  # 每90帧检测一次
        # 跟踪相关变量
        tracking_objects = {}  # 存储跟踪ID和边界框
        #上一帧位置
        prev_obj1_box, prev_obj2_box = None, None
        stable_positions = []  # 用于保存稳定检测位置的历史记录
        max_history = 5  # 保存的最大历史位置数
        detection_fail_count = 0  # 连续检测失败计数
        max_fail_threshold = 3  # 最大允许连续失败次数
        # 打开摄像头并清帧
        cap = cv2.VideoCapture(1)  # 使用摄像头1
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        print("摄像头打开成功")
        for _ in range(30):
            cap.grab()
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

                obj1_box, obj2_box,detected = new_robust_dual_object_detection(frame_rgb, prev_obj1_box,
                                                                               prev_obj2_box, object1_name,
                                                                               object2_name,
                                                                               scale_factor=2)
                # if not detected:
                #     max_fail_threshold+= 1
                # if max_fail_threshold >= max_fail_threshold:#连续三帧检测不到
                #     pass
                print("obj1_box, obj2_box, ",obj1_box, obj2_box)
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
                    print("&&&&&&&&开始深度移动一次&&&&&&&&")
                    two_calculate_xy.move_controller_in_depth_only(0.3)
                    for _ in range(10):
                        cap.grab()
                else:
                    pass

                """抑制iou的移动"""
                iou_result = two_calculate_xy.calculate_distance_and_movement(obj1_box, obj2_box, tolerance=40)
                x,y =iou_result["direction_x"],iou_result["direction_y"]
                print("frame_count","need_move",iou_result["need_move"],iou_result["direction_x"],iou_result["direction_y"])
                if iou_result["need_move"] and not depth_dict["need_move"]:
                    print(f"&&&&开始平面移动:{x}{y}&&&&&")
                    #iou得上下和自然上下相反，所以取负
                    distance = calculate_diagonal_length_in_meters(x,y)
                    two_calculate_xy.move_controller_by_local_direction(iou_result["direction_x"],-iou_result["direction_y"],0,distance*1.2)
                    # 清帧
                    for _ in range(10):
                        cap.grab()
                    continue
                else:
                    #确认iou抑制
                    pass
                if not depth_dict["need_move"] and not iou_result["need_move"]:
                    print("结束移动")
                    break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


def calculate_diagonal_length_in_meters(dx, dy):
    # 计算对角线长度（欧几里得距离）
    diagonal_length = ((dx) ** 2 + (dy) ** 2) ** 0.5

    # 应用系数转换为米
    coefficient = 0.3  # 坐标系单位到米的转换系数
    diagonal_length_in_meters = diagonal_length * coefficient

    return diagonal_length_in_meters


def is_position_consistent(prev_box, new_box, move_history, max_deviation_factor=2):
    """基于中心漂移和尺寸变化判断一致性，prev_box, new_box: (x, y, w, h)"""
    x0, y0, w0, h0 = prev_box
    x1, y1, w1, h1 = new_box

    # 计算中心点
    cx0, cy0 = x0 + w0 / 2, y0 + h0 / 2
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2

    # 计算当前移动距离
    current_dist = math.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)

    # 评判位置一致性
    if prev_box is None:
        position_consistent = True
    elif move_history == 10000:  # 没有历史移动记录
        position_consistent = True
    else:
        position_consistent = current_dist <= move_history * max_deviation_factor

    # 评判尺寸一致性
    area1 = w0 * h0
    area2 = w1 * h1

    if area1 > 0:
        area_ratio = area2 / area1
        # 不对称尺寸判断：允许变小到40%，但只允许变大到120%
        size_consistent = 0.1 <= area_ratio <= 1.2
    else:
        size_consistent = False

    # 同时满足位置和尺寸一致才返回True
    is_consistent = position_consistent and size_consistent

    # 输出详细信息
    print(f"位置一致: {position_consistent} (距离={current_dist:.2f}, 阈值={move_history * max_deviation_factor if move_history else 'N/A'})")
    print(f"尺寸一致: {size_consistent} (面积比={area_ratio:.2f} if area1 > 0 else 'N/A')")

    return is_consistent, current_dist



if __name__ == "__main__":
    main_3()
