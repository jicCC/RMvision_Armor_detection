#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "types.hpp"
// 一根灯条。
// top / bottom 是灯条中轴线的两个端点，也就是"两个角点"，
// 后面拼装甲板四个角点的时候直接用它们。
struct LightBar 
{
    cv::Point2f center;      // 灯条中心
    cv::Point2f top;         // 中心线的上端点（= 上边界中点，也就是"角点"）
    cv::Point2f bottom;      // 中心线的下端点
    float length{0.f};       // 灯条长度（长轴方向）
    float width{0.f};        // 灯条宽度
    float angle{0.f};        // 长轴与竖直方向的夹角（度）
};
// 传统视觉方案：自己找灯条、自己配对、自己认数字。
// 和神经网络方案的作用完全一样（都是给出一串 Detection），
// 所以后面的 PnP 解算和画图代码能两边共用。
class TraditionalDetector
{
public:
    // is_blue：true 处理蓝方灯条，false 处理红方。搞反了 mask 会全黑
    // digit_model_path：数字网络 mlp.onnx 的路径
    TraditionalDetector(bool is_blue, const std::string& digit_model_path);
    std::vector<Detection> detect(const cv::Mat& img);
    void draw(cv::Mat& img, const std::vector<Detection>& dets);
    // 是否要求"两灯条之间必须识别出数字"才判定为装甲板。
    // 真实场景（人员走动、杂物多）建议开启；仿真图片中远处小目标数字识别会失败，可关闭。
    // 这一开关对应讲义里"判断两根灯条中间是否有数字"那一步
    void setRequireDigit(bool v) { require_digit_ = v; }
private:
    cv::Mat                binarize(const cv::Mat& img);                      // 二值化
    std::vector<LightBar>  findLightBars(const cv::Mat& mask);                // 找灯条
    std::vector<Detection> matchLightBars(const cv::Mat& img, const std::vector<LightBar>& bars); // 配对
    int                    recognizeDigit(const cv::Mat& img, const LightBar& left, const LightBar& right);
    bool is_blue_;
    bool require_digit_{false};       // 默认不强制
    // ---------- OpenVINO：数字网络 ----------
    ov::Core           digit_core_;
    ov::CompiledModel  digit_model_;
    ov::InferRequest   digit_request_;
};