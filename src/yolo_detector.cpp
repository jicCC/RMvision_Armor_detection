#include "yolo_detector.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

using namespace std;

namespace {

// 网络的输入输出规格
constexpr int   INPUT_W     = 640;    // 输入宽
constexpr int   INPUT_H     = 480;    // 输入高
constexpr int   NUM_ANCHORS = 6300;   // 三个尺度的预测点加起来：80*60 + 40*30 + 20*15
constexpr int   NUM_CLASSES = 9;      // label.txt 里的 9 个类别
constexpr int   KPT_COUNT   = 4;      // 每个目标 4 个角点
constexpr int   KPT_STRIDE  = 2;      // 每个角点占两个数（x, y）

// 输出的 21 个通道是怎么分的
constexpr int   CH_CLASS_BEGIN = 4;   // 通道 4~12 是 9 个类别分数
constexpr int   CH_KPT_BEGIN   = 13;  // 通道 13~20 是 4 个角点的坐标

constexpr float CONF_THRESHOLD = 0.4f;
constexpr float NMS_THRESHOLD  = 0.5f;
constexpr float NORM_SCALE     = 1.0f / 255.0f;

constexpr int    LINE_THICK  = 2;
constexpr int    TEXT_OFFSET = 5;
constexpr double FONT_SCALE  = 0.5;

// 关键点的顺序映射。
// 模型原始的 4 个点是 0=左上, 1=左下, 2=右下, 3=右上，
// 而我们在整个工程里统一用 0=左上, 1=右上, 2=右下, 3=左下。
// 这个顺序是用检测结果投影到图上对出来的，不是文档里写的，
// 所以如果换了模型一定要重新确认一遍
constexpr int KPT_MAP[KPT_COUNT] = {0, 3, 2, 1};

} // namespace

// 构造函数：加载模型。这一句挺慢的（几百毫秒），所以只做一次
YoloDetector::YoloDetector(const std::string& model_path)
{
    cout << "正在加载模型: " << model_path << endl;

    shared_ptr<ov::Model> model = core.read_model(model_path);
    compiled_model = core.compile_model(model, "CPU");
    infer_request  = compiled_model.create_infer_request();
}

// 识别：一张图进去，一串检测结果出来。
// 流程就是 预处理 → 推理 → 逐个预测点解码 → NMS 去重
std::vector<Detection> YoloDetector::detect(const cv::Mat& img)
{
    // 预处理：缩放到网络的输入尺寸，转成 RGB，像素值除以 255。
    // 测试图是 960×720、视频是 1440×1080，宽高比都是 4:3，和 640×480 一样，
    // 所以直接缩放不会变形
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, NORM_SCALE, cv::Size(INPUT_W, INPUT_H),
                           cv::Scalar(0, 0, 0), true, false);

    // 原图宽度是输入宽度的多少倍，坐标还原成原图坐标时就乘回去。
    // 960 宽是 1.5，1440 宽是 2.25
    const float inv_scale = img.cols / float(INPUT_W);

    ov::Tensor input_tensor(ov::element::f32, {1, 3, INPUT_H, INPUT_W}, blob.ptr<float>());
    infer_request.set_input_tensor(input_tensor);

    infer_request.infer();
    const float* data = infer_request.get_output_tensor().data<const float>();
    // 输出是 [1, 21, 6300]，内存里是"通道优先"排列的，
    // 想取"第 c 个数、第 i 个预测点"就要写 data[c * 6300 + i]，不能写 data[i * 21 + c]
    //
    // 这 21 个通道分成三组：
    //   0~3   框的回归量。这套模型里这几个数没有意义（实测最大能到 0.887、
    //         最小是 2.5e-14，完全不像坐标），所以不能用它当框
    //   4~12  9 个类别的分数，也就是装甲板编号
    //   13~20 4 个角点，每两个数一个点

    // 解码：把 6300 个预测点逐个看一遍，分数够高的留下来
    std::vector<cv::Rect>  boxes;
    std::vector<float>     confidences;
    std::vector<Detection> cand;

    for (int i = 0; i < NUM_ANCHORS; ++i)
    {
        // 只在类别那 9 个通道里找最大值。
        // 千万不要把 0~3 也算进来——那几个数能到 0.8 以上，会把真正的类别分压下去
        int   cls   = 0;
        float score = data[CH_CLASS_BEGIN * NUM_ANCHORS + i];
        for (int c = 1; c < NUM_CLASSES; ++c) {
            float s = data[(CH_CLASS_BEGIN + c) * NUM_ANCHORS + i];
            if (s > score) { score = s; cls = c; }
        }
        if (score < CONF_THRESHOLD) continue;

        // 取 4 个角点，乘回缩放倍数换成原图坐标，按统一的顺序存
        Detection d;
        d.cls   = cls;
        d.score = score;
        for (int n = 0; n < KPT_COUNT; ++n) {
            int k = KPT_MAP[n];
            float kx = data[(CH_KPT_BEGIN + KPT_STRIDE * k) * NUM_ANCHORS + i] * inv_scale;
            float ky = data[(CH_KPT_BEGIN + 1 + KPT_STRIDE * k) * NUM_ANCHORS + i] * inv_scale;
            d.kpt.push_back(cv::Point2f(kx, ky));
        }

        // 因为模型的框通道不能用，所以拿四个角点自己算一个外接矩形。
        // NMS 需要矩形才能算重叠面积
        float x_min = d.kpt[0].x, x_max = d.kpt[0].x;
        float y_min = d.kpt[0].y, y_max = d.kpt[0].y;
        for (int k = 1; k < 4; ++k) {
            x_min = std::min(x_min, d.kpt[k].x);
            x_max = std::max(x_max, d.kpt[k].x);
            y_min = std::min(y_min, d.kpt[k].y);
            y_max = std::max(y_max, d.kpt[k].y);
        }
        d.box = cv::Rect2f(x_min, y_min, x_max - x_min, y_max - y_min);

        boxes.emplace_back(int(x_min), int(y_min),
                           int(x_max - x_min), int(y_max - y_min));
        confidences.push_back(score);
        cand.push_back(d);
    }

    // NMS：同一个目标在三个尺度上都会出预测点，会有好几个重叠的框，这里去掉重复的
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Detection> results;
    for (int idx : indices) results.push_back(cand[idx]);
    return results;
}

void YoloDetector::draw(cv::Mat& img, const std::vector<Detection>& dets)
{
    for (const auto& d : dets)
    {
        // 4 个角点连成四边形，用绿色，和传统方案的黄框区分
        std::vector<cv::Point> pts;
        for (const auto& p : d.kpt) pts.push_back(cv::Point(int(p.x), int(p.y)));
        cv::polylines(img, pts, true, cv::Scalar(0, 255, 0), LINE_THICK);

        // 标签放在框的【上方】。传统方案的标签在框下方，一上一下就不会重叠
        std::string text = cv::format("cls=%d  %.2f", d.cls, d.score);
        cv::putText(img, text, cv::Point(int(d.box.x), int(d.box.y) - TEXT_OFFSET),
                    cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, cv::Scalar(0, 255, 0), 1);
    }
}