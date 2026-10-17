#include "yolo_detector.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

using namespace std;

// ================= 构造函数：加载模型 =================
YoloDetector::YoloDetector(const std::string& model_path)
{
    cout << "正在加载模型: " << model_path << endl;

    shared_ptr<ov::Model> model = core.read_model(model_path);
    compiled_model = core.compile_model(model, "CPU");
    infer_request  = compiled_model.create_infer_request();
}

// ================= 识别：图 -> 检测结果列表 =================
std::vector<Detection> YoloDetector::detect(const cv::Mat& img)
{
    const int   NUM_ANCHORS    = 6300;   // 80*60 + 40*30 + 20*15
    const int   NUM_CLASSES    = 9;      // label.txt 有 9 类
    const float CONF_THRESHOLD = 0.4f;
    const float NMS_THRESHOLD  = 0.5f;

    // ---------- 1. 预处理：缩到 640x480，转 RGB，除 255 ----------
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(640, 480),
                           cv::Scalar(0, 0, 0), true, false);

    // 原图宽度是 640 的多少倍，坐标还原时就乘回去
    // 图片 960 宽 -> 1.5；视频 1440 宽 -> 2.25
    const float inv_scale = img.cols / 640.0f;

    ov::Tensor input_tensor(ov::element::f32, {1, 3, 480, 640}, blob.ptr<float>());
    infer_request.set_input_tensor(input_tensor);

    // ---------- 2. 推理 ----------
    infer_request.infer();
    const float* data = infer_request.get_output_tensor().data<const float>();
    // data 的排布是 [1, 21, 6300]：
    //   下标 0..3   = 框（这套模型里是废数据，不用）
    //   下标 4..12  = 9 个类别分数
    //   下标 13..20 = 4 个角点 (x1,y1,x2,y2,x3,y3,x4,y4)
    // 取"第 c 个数、第 i 个 anchor"就是 data[c * 6300 + i]

    // ---------- 3. 解码 ----------
    std::vector<cv::Rect>  boxes;
    std::vector<float>     confidences;
    std::vector<Detection> cand;

    for (int i = 0; i < NUM_ANCHORS; ++i)
    {
        // (1) 只在下标 4..12 里找最大的分数，并记下是第几个赢的
        int   cls   = 0;
        float score = data[4 * NUM_ANCHORS + i];
        for (int c = 1; c < NUM_CLASSES; ++c) {
            float s = data[(4 + c) * NUM_ANCHORS + i];
            if (s > score) { score = s; cls = c; }
        }
        if (score < CONF_THRESHOLD) continue;

        // (2) 取 4 个角点，映射回原图坐标（存小数）
        //     模型原始关键点顺序：0=左上, 1=左下, 2=右下, 3=右上
        //     统一约定（canonical，见 types.hpp）：0=左上, 1=右上, 2=右下, 3=左下
        //     所以输出第 n 个角点时要取模型的第 MAP[n] 号关键点
        const int MAP[4] = {0, 3, 2, 1};

        Detection d;
        d.cls   = cls;
        d.score = score;
        for (int n = 0; n < 4; ++n) {
            int k = MAP[n];
            float kx = data[(13 + 2 * k) * NUM_ANCHORS + i] * inv_scale;
            float ky = data[(14 + 2 * k) * NUM_ANCHORS + i] * inv_scale;
            d.kpt.push_back(cv::Point2f(kx, ky));
        }

        // (3) 用 4 个角点算外接矩形，给 NMS 用
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

    // ---------- 4. NMS 去掉重复的框 ----------
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Detection> results;
    for (int idx : indices) results.push_back(cand[idx]);
    return results;
}

// ================= 画图 =================
void YoloDetector::draw(cv::Mat& img, const std::vector<Detection>& dets)
{
    for (const auto& d : dets)
    {
        // 4 个角点连成四边形
        std::vector<cv::Point> pts;
        for (const auto& p : d.kpt) pts.push_back(cv::Point(int(p.x), int(p.y)));
        cv::polylines(img, pts, true, cv::Scalar(0, 255, 0), 1);

        // 先显示类别序号，别去查 label.txt
        std::string text = cv::format("cls=%d  %.2f", d.cls, d.score);
        cv::putText(img, text, cv::Point(int(d.box.x), int(d.box.y) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}