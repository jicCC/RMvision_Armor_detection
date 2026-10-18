#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "types.hpp"
// 神经网络方案：用 armor.onnx 直接检测装甲板。
// 这个网络输出的 9 个类别就是装甲板编号，所以在它这里"认数字"是顺带的，
// 不像传统方案那样还得另外接一个数字网络。
class YoloDetector
{
public:
    // model_path 一般是 "data/armor.onnx"
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