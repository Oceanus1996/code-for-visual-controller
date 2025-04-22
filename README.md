# code-for-visual-controller
一、环境准备
1.下载 代码

主要依赖：transformers, torch, opencv-python, diffimg, filterpy 等。

2.控制器调试模块

在 GroundingDINO/two_calculate_xy 和 indle 模块中，封装了移动控制器到初始偏移、深度移动和平面移动的函数。

确保在运行前已完成 SteamVR/OpenVR 的初始化并连接好 VR 头盔与手柄。

3.安装 OBS 用于摄像头捕捉

从 OBS 官网 下载并安装最新版本。

在 OBS 中添加“显示捕获”或“视频捕获设备”以获取 cv2.VideoCapture 对应的摄像头索引（如 1）。

4.下载 MiDaS 与相关模型

pip install midas torch torchvision timm

或者根据项目 indle.midas_setup() 自动下载并放置模型文件。
