import warnings
import torch
import cv2
import numpy as np

# 全局变量
midas = None
transform = None
device = None

def midas_setup():
    """
    初始化 MiDaS 模型和预处理 transform，结果保存在全局变量里。
    """
    global midas, transform, device

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    midas = torch.hub.load(
        "intel-isl/MiDaS", "DPT_Hybrid", source="github"
    ).to(device).eval()
    # 加载预处理
    transforms = torch.hub.load(
        "intel-isl/MiDaS", "transforms", source="github"
    )
    transform = transforms.dpt_transform


def compare_depth(box1, box2, frame, depth_threshold=15):
    """
    对一帧图像 frame，比较两个框 box1/box2 的平均深度，以及它们的交集平均深度，
    并根据 depth_threshold 决定是否需要移动。

    Args:
        box1, box2: (x, y, w, h) 两个检测框
        frame: BGR 图像 (H, W, 3)
        depth_threshold: 深度差阈值，默认为 0.02

    Returns:
        dict 包含键:
          'd1', 'd2', 'd_inter' : 三个平均深度值
          'need_move'           : bool, 是否需要移动
          'd_diff'              : d1 - d2   (正表示 box1 在后面更远)
    """
    # 1) MiDaS 推理
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).to(device)           # [1,3,H,W]
    with torch.no_grad():
        pred = midas(inp)
    depth_map = pred.squeeze().cpu().numpy()  # [h',w']
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # 2) 裁出两个 ROI
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    roi1 = depth_map[y1:y1+h1, x1:x1+w1]
    roi2 = depth_map[y2:y2+h2, x2:x2+w2]

    # 3) 计算平均深度
    d1 = float(np.nanmean(roi1))
    d2 = float(np.nanmean(roi2))


    # 4) 计算交集区域平均深度
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    if xi2 > xi1 and yi2 > yi1:
        roi_i = depth_map[yi1:yi2, xi1:xi2]
        d_inter = float(np.nanmean(roi_i))
    else:
        d_inter = None

    print("计算平均深度","d1",d1,"d2",d2,"d_inter",d_inter)

    # 5) 判断是否需要移动
    dont_need_move = abs(d1 - d2) <= depth_threshold or d2 >= d1 * 1.2
    # 然后取反
    need_move = not dont_need_move
    d_diff = d1 - d2

    return {
        'd1': d1,
        'd2': d2,
        'd_inter': d_inter,
        'need_move': need_move,
        'd_diff': d_diff
    }
