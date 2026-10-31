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

        // 若有观测反解的方位角，就用它初始化（init 内部假设这是板 0，φ=0，
        // 故可直接传入 θ_j）；否则退回猜 0
        const double th0 = meas[0].hasTheta ? meas[0].thetaMeas : 0.0;

        ekf_.init(p.at<double>(0), p.at<double>(1), p.at<double>(2),
                  th0, PLATE_RADIUS, 0.10);

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
        pushNis(false);
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
        pushNis(false);

        if (badConverge())
        {
            ++branchTried_;
            if (tryReanchor(meas))
            {
                ++branchFixes_;
                miss_ = 0;
                return matched_;
            }
            nisWin_.clear();        // 试过就清空，避免每帧重试
        }

        if (miss_ > kLost) inited_ = false; //同上，太久没有观测，下次需要重建
        return -1;
    }


    //更新
    pushNis(true);
    ekf_.correct(best_j, meas[best_m].pos, R_);
    matched_ = best_j;
    miss_    = 0;
    return best_j;

}


void ArmorTracker::pushNis(bool ok)
{
    nisWin_.push_back(ok);
    while (static_cast<int>(nisWin_.size()) > kWindow) nisWin_.pop_front();
}


bool ArmorTracker::badConverge() const
{
    if(static_cast<int>(nisWin_.size()) < kMinForJudge) //窗口太短不判断
        return false;

    int fails = 0;
    for (bool ok : nisWin_) 
        if (!ok) 
            ++fails;

    return fails >= kFailRatio * static_cast<double>(nisWin_.size());
}


bool ArmorTracker::tryReanchor(const std::vector<PlateMeas>& meas)
{
    if (meas.empty())
        return false;

    // 当前分支下，当前帧的最小 NIS
    double bestNow = 1e18;
    for (const auto& m : meas)
        for (int j = 0; j < PLATE_NUM; ++j)
        {
            const double d2 = nis(j, m.pos);
            if (d2 < bestNow) bestNow = d2;
        }


    const double cand[3] = { M_PI, M_PI/2.0, -M_PI/2.0 }; //候选朝向

        double bestNew   = bestNow;
    double bestShift = 0.0;
    int    bestM = -1, bestJ = -1;
    

    //对另外三个朝向都试一下，看哪个好
    for (int c = 0; c < 3; ++c)
    {
        ekf_.shiftTheta(cand[c]);

        for (int m = 0; m < static_cast<int>(meas.size()); ++m)
            for (int j = 0; j < PLATE_NUM; ++j)
            {
                const double d2 = nis(j, meas[m].pos);
                if (d2 < bestNew) { bestNew = d2; bestShift = cand[c]; bestM = m; bestJ = j; }
            }

        ekf_.shiftTheta(-cand[c]);          
    }

    //都不够好就不切换
    if (bestShift == 0.0)         return false;
    if (bestNew > bestNow * 0.5)  return false;    
    if (bestNew > gate_)          return false;

    ekf_.shiftTheta(bestShift);
    ekf_.correct(bestJ, meas[bestM].pos, R_);
    matched_ = bestJ;
    nisWin_.clear();                             

    std::printf("  [tracker] 分支重收敛: theta %+.1f deg, NIS %.2f -> %.2f, board %d\n",
                bestShift * 180.0 / M_PI, bestNow, bestNew, bestJ);
    return true;
}