#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "ekf.hpp"

// 一次观测（世界系下的装甲板中心）
struct PlateMeas
{
    cv::Mat pos;        
    int     digit{0};
    double  score{0.0};
};

class ArmorTracker
{
public:
    ArmorTracker(double gate = 11.34, double sigma = 0.10);

    // 喂一帧，t 是该帧时间戳（秒）。返回匹配到的板号，-1 表示没匹配上
    int update(const std::vector<PlateMeas>& meas, double t);

    const Ekf& ekf() const { return ekf_; }

    int  matched()   const { return matched_; }
    int  missCount() const { return miss_; }
    bool inited()    const { return inited_; }

private:
    double nis(int j, const cv::Mat& z) const;

    static constexpr int kLost = 10;   // 连续多少帧没匹配上算丢失

    Ekf      ekf_;
    cv::Mat  R_;                      
    double   gate_;
    double   last_t_{-1.0};
    int      matched_{-1};
    int      miss_{0};
    bool     inited_{false};
};