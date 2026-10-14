#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

struct Detection 
{
    int   cls;                      // 模型认为的类别序号（先别查 label.txt）
    float score;                    // 这个类别的分数
    cv::Rect2f box;                 // 框，给 NMS 用
    std::vector<cv::Point2f> kpt;   // 4 个角点，原图坐标（PnP 要用的）
};

class YoloDetector
{
public:
    YoloDetector(const std::string& model_path);

    // 识别：返回"找到了什么"，不画图
    std::vector<Detection> detect(const cv::Mat& img);

    // 画图：给它图 + 结果列表
    void draw(cv::Mat& img, const std::vector<Detection>& dets);

private:
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
};