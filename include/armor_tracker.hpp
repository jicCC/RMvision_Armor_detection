#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "ekf.hpp"
#include <deque>

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

    int  branchTried() const { return branchTried_; }
    int  branchFixes() const { return branchFixes_; }

private:
    double nis(int j, const cv::Mat& z) const;

    void pushNis(bool ok);                                  // 记录一帧 NIS 结果（借鉴sp_25vision开源）
    bool badConverge() const;                               // 窗口内失败率是否过高（借鉴sp_25vision开源）
    bool tryReanchor(const std::vector<PlateMeas>& meas);   // 尝试分支平移（借鉴sp_25vision开源）

    static constexpr int kLost = 10;   // 连续多少帧没匹配上算丢失
    static constexpr int    kWindow      = 20;     // NIS 滑窗长度（借鉴sp_25vision开源）
    static constexpr double kFailRatio   = 0.40;   // 窗口内失败比例阈值（借鉴sp_25vision开源）
    static constexpr int    kMinForJudge = 10;     // 窗口最短长度（借鉴sp_25vision开源）


    std::deque<bool> nisWin_;      // 每帧的 NIS 是否"失败"（借鉴sp_25vision开源）
    int  branchTried_{0};          // 尝试重收敛的次数（借鉴sp_25vision开源）
    int  branchFixes_{0};          // 重收敛成功的次数（借鉴sp_25vision开源）

    Ekf      ekf_;
    cv::Mat  R_;                      
    double   gate_;
    double   last_t_{-1.0};
    int      matched_{-1};
    int      miss_{0};
    bool     inited_{false};
};