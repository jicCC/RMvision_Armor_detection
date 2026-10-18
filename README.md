# RM 装甲板识别与位姿解算

## 一、环境要求

| 项        | 版本 / 说明                                    |
| -------- | ------------------------------------------ |
| 系统       | Ubuntu 20.04 及以上（本工程在 WSL2 的 Ubuntu 下开发验证） |
| 编译器      | 支持 C++17                                   |
| CMake    | ≥ 3.10                                     |
| OpenCV   | ≥ 4.5（开发时用 4.10.0，需含）                      |
| OpenVINO | ≥2024.x（开发时用2026.4.1）                      |

`CMakeLists.txt` 里用的是 `find_package(OpenCV REQUIRED)` 和 `find_package(OpenVINO REQUIRED)`，
所以两个库都需要能被 CMake 找到（OpenVINO 一般需要先 `source setupvars.sh`）。

---

## 二、数据准备

把下面这些文件放到工程的 `data/` 目录下：

```
data/
├── armor.onnx       # 装甲板检测网络（YOLOv8-pose，输入 1x3x480x640，输出 1x21x6300）
├── mlp.onnx         # 数字识别网络（输入 1x20x28x1，输出 9 类）
├── test1.jpg        # 测试图片，960x720
├── test2.jpg
├── test3.jpg
└── video.avi        # 测试视频，1440x1080、30fps、531 帧（原文件名 2026-05-20_23-57-31.avi）
```

---

## 三、构建

```bash
cd RMvision_2ndHomework
cmake -S . -B build
cmake --build build -j
```

如果 OpenVINO 找不到，先执行它的环境脚本再构建：

```bash
source /opt/intel/openvino/setupvars.sh
```

---

## 四、运行

**必须在工程根目录下运行**，因为代码里的数据路径都是相对路径（`data/xxx`）：

```bash
./build/test_opencv
```

运行后会依次做三件事：

1. 打印视频的基本信息（分辨率、帧数、帧率）
2. 处理三张测试图片：两种方案各解一次位姿，输出误差表，并保存三张结果图
3. 处理视频：两种方案各输出一个标注视频

全程大约 30 秒（主要花在视频上，531 帧 × 2 遍）。

### 输出文件

| 文件                                                              | 内容                     |
| --------------------------------------------------------------- | ---------------------- |
| `result_0_trad.jpg` / `result_1_trad.jpg` / `result_2_trad.jpg` | 只画传统方案的黄框              |
| `result_0_yolo.jpg` / `result_1_yolo.jpg` / `result_2_yolo.jpg` | 只画神经网络的绿框              |
| `result_0_cmp.jpg` / `result_1_cmp.jpg` / `result_2_cmp.jpg`    | 两种方案的框叠在同一张图上，用于对比角点位置 |
| `result_video_trad.avi`                                         | 传统方案的标注视频              |
| `result_video_yolo.avi`                                         | 神经网络方案的标注视频            |

（`result_0` 对应 `test1.jpg`，依次类推）

标注视频里两种方案用颜色区分：**传统方案全黄、神经网络全绿**，
框旁边会写装甲板编号`cls=__`，并叠加一行动解算出来的距离（如 `1.45m`）。

---

## 五、文件结构

```
RMvision_2ndHomework/
├── CMakeLists.txt
├── README.md
├── data/                      # 模型、测试图片、测试视频（见第二节）
├── include/
│   ├── types.hpp              # 两种方案共用的 Detection 结构体
│   ├── yolo_detector.hpp      # 神经网络方案
│   ├── traditional_detector.hpp  # 传统方案（含 LightBar 结构体）
│   └── pnp_solver.hpp         # PnP 解算器
└── src/
    ├── main.cpp               # 流程编排：图片 → 误差表 → 视频
    ├── yolo_detector.cpp      # 网络推理 + 输出解码 + 画框
    ├── traditional_detector.cpp  # 二值化 / 找灯条 / 配对 / 数字识别
    └── pnp_solver.cpp         # solvePnP + 重投影误差
```

---

## 六、其他需要注意的点

### 1. 两套相机内参不能混用

素材里的测试图片和视频来自**不同的相机**：

|      | 分辨率       | fx      | fy      | cx     | cy     | 畸变                                            |
| ---- | --------- | ------- | ------- | ------ | ------ | --------------------------------------------- |
| 测试图片 | 960×720   | 869.12  | 869.12  | 480.00 | 360.00 | 无                                             |
| 素材视频 | 1440×1080 | 2362.28 | 2336.19 | 679.18 | 529.07 | k1=−0.0986, k2=0.1117, p1=−0.0038, p2=−0.0020 |

`fx` 差了 **2.7 倍**。混用不会报错，但解出来的距离会正好差 2.7 倍。
代码里在 `main.cpp` 顶部把两套参数分别定义成 `IMG_*` 和 `VID_*`，并各自建了一个 `PnpSolver`。

### 2. 装甲板颜色要选对

传统方案靠"目标通道减去另两个通道的最大值"来分离灯条，所以必须知道是红方还是蓝方。

```cpp
TraditionalDetector trad_img(true,  "data/mlp.onnx");   // true  = 蓝方（图片）
TraditionalDetector trad_vid(false, "data/mlp.onnx");   // false = 红方（视频）
```


