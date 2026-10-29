#include "armor_tracker.hpp"
#include <cmath>
#include <cstdio>

ArmorTracker::ArmorTracker(double gate, double sigma)
    : gate_(gate)
{
    // 世界系下：x、y 用 sigma，z（深度方向的影响）给更大
    R_ = cv::Mat::zeros(3, 3, CV_64F);
    R_.at<double>(0, 0) = sigma * sigma;              // 0.01
    R_.at<double>(1, 1) = sigma * sigma;              // 0.01
    R_.at<double>(2, 2) = (sigma * 2.5) * (sigma * 2.5);   // 0.0625
}


// 某一帧的观测 z 如果属于"板 j"，它的 NIS 是多少（NIS是要识别哪个装甲板的判据）
double ArmorTracker::nis(int j, const cv::Mat& z) const
{
    const cv::Mat xs = ekf_.state();
    const cv::Mat H  = ekf_.H(j, xs);
    const cv::Mat hx = ekf_.h(j, xs);
    const cv::Mat e  = z - hx;

    cv::Mat S = H * ekf_.cov() * H.t() + R_;
    S = 0.5 * (S + S.t());

    cv::Mat Sinv;
    cv::invert(S, Sinv, cv::DECOMP_CHOLESKY);

    const cv::Mat d2m = e.t() * Sinv * e;    // ← 先落到一个 cv::Mat 里
    return d2m.at<double>(0);
}



int ArmorTracker::update(const std::vector<PlateMeas>& meas, double t)
{
    //如果还没初始化
    if (!inited_)
    {
        if (meas.empty()) return -1;

        const cv::Mat& p = meas[0].pos;
        // 把这次观测当成"板 0"，θ 先猜 0，靠后续观测修正
        ekf_.init(p.at<double>(0), p.at<double>(1), p.at<double>(2),
                  0.0, PLATE_RADIUS, 0.10);

        inited_  = true;
        last_t_  = t;
        matched_ = 0;
        miss_    = 0;
        return 0;
    }

    //预测
    double dt = t - last_t_;
    if (dt < 0.0)  dt = 0.0;
    if (dt > 0.30) dt = 0.30;          // 时间跨度太大，只按照0.3s推，防止误差过大
    if (dt > 0.0)  ekf_.predict(dt);
    last_t_ = t;


    //如果这一帧没有观测
    if (meas.empty())
    {
        ++miss_;
        matched_ = -1;
        if (miss_ > kLost) inited_ = false;      //太久没有观测，下次需要重建
        return -1;
    }


    //匹配：通过枚举观测和板号找NIS最小的
    double best_d2 = 1e18;
    int    best_m  = -1;
    int    best_j  = -1;

    for (int m = 0; m < static_cast<int>(meas.size()); ++m)
    {
        for (int j = 0; j < PLATE_NUM; ++j)
        {
            const double d2 = nis(j, meas[m].pos);
            if (d2 < best_d2) { best_d2 = d2; best_m = m; best_j = j; }
        }
    }


    //门限检验：如果最小的NIS都很大，就直接判定为观测不可用，等同于未观测的效果
    if (best_d2 > gate_)
    {
        ++miss_;
        matched_ = -1;
        if (miss_ > kLost) inited_ = false;    //同上，太久没有观测，下次需要重建
        return -1;
    }


    //更新
    ekf_.correct(best_j, meas[best_m].pos, R_);
    matched_ = best_j;
    miss_    = 0;
    return best_j;

}