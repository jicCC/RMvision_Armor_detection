#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "types.hpp"

// 一根灯条
struct LightBar 
{
    cv::Point2f center;      // 灯条中心
    cv::Point2f top;         // 中心线的上端点（= 上边界中点，也就是"角点"）
    cv::Point2f bottom;      // 中心线的下端点
    float length{0.f};       // 灯条长度（长轴方向）
    float width{0.f};        // 灯条宽度
    float angle{0.f};        // 长轴与竖直方向的夹角（度）
};

class TraditionalDetector
{
public:
    TraditionalDetector(bool is_blue, const std::string& digit_model_path);    // true = 蓝方灯条，false = 红方

    std::vector<Detection> detect(const cv::Mat& img);

    void draw(cv::Mat& img, const std::vector<Detection>& dets);

private:
    cv::Mat                binarize(const cv::Mat& img);                      // 二值化
    std::vector<LightBar>  findLightBars(const cv::Mat& mask);                // 找灯条
    std::vector<Detection> matchLightBars(const cv::Mat& img, const std::vector<LightBar>& bars); // 配对
    int                    recognizeDigit(const cv::Mat& img, const LightBar& left, const LightBar& right);

    bool is_blue_;

    // ---------- OpenVINO：数字网络 ----------
    ov::Core           digit_core_;
    ov::CompiledModel  digit_model_;
    ov::InferRequest   digit_request_;
};