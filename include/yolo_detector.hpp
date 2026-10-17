#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "types.hpp"




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