#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
// 两种方案共同的输出格式。有了它，后面的 PnP 解算和画图代码就能两种方案共用一份，
// 不用写两套。
struct Detection 
{
    int   cls;                      // 类别序号（神经方案=装甲板网络输出；传统方案=数字网络输出）
    float score;                    // 这个类别的置信度
    cv::Rect2f box;                 // 外接矩形，给 NMS 用
    std::vector<cv::Point2f> kpt;   // 4 个角点，原图坐标（PnP要用）
                                    // 顺序固定为 左上 → 右上 → 右下 → 左下（顺时针），
                                    // 必须和 pnp_solver 里物点的顺序一致，不然解出来的位姿是歪的
};