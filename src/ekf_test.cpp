// ============================================================================
//  ekf_test.cpp —— 不依赖视频的 EKF 全流程验证
//  跑一次约 0.1 秒
// ============================================================================
#include "ekf.hpp"
#include "imu.hpp"
#include "armor_tracker.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>

int main()
{
    // ======================================================================
    //  阶段 1：检查 h()  —— 四块板应该排成十字
    // ======================================================================
    printf("=== 阶段 1：检查 h() ===\n\n");

    cv::Mat x = cv::Mat::zeros(DIM, 1, CV_64F);
    x.at<double>(ZA)    = 0.25;
    x.at<double>(R_ARM) = PLATE_RADIUS;

    Ekf ekf;

    for (int j = 0; j < PLATE_NUM; ++j)
    {
        cv::Mat p = ekf.h(j, x);
        printf("  板%d: (%7.3f, %7.3f, %7.3f)\n", j,
               p.at<double>(0), p.at<double>(1), p.at<double>(2));
    }

    // ======================================================================
    //  阶段 2：验证 predict()
    // ======================================================================
    printf("\n=== 阶段 2：验证 predict() ===\n");

    Ekf e2;
    {
        const double a1 = e2.cov().at<double>(CX, CX);
        const double a2 = e2.cov().at<double>(VX, VX);
        printf("  预测前: P(0,0)=%.6e  P(1,1)=%.6e\n", a1, a2);
    }
    e2.predict(0.0415);
    {
        const double a1 = e2.cov().at<double>(CX, CX);
        const double a2 = e2.cov().at<double>(VX, VX);
        printf("  预测后: P(0,0)=%.6e  P(1,1)=%.6e\n", a1, a2);
    }

    // ======================================================================
    //  阶段 3：验证 correct()
    // ======================================================================
    printf("\n=== 阶段 3：验证 correct() ===\n");

    Ekf e3;
    e3.predict(0.0415);

    const double before_cx = e3.state().at<double>(CX);
    const double before_vx = e3.state().at<double>(VX);
    const double before_th = e3.state().at<double>(THETA);
    const double before_r  = e3.state().at<double>(R_ARM);
    const double before_P  = e3.cov().at<double>(CX, CX);

    cv::Mat Rm = cv::Mat::eye(3, 3, CV_64F) * 0.01;      // σ = 0.1 m
    cv::Mat z  = (cv::Mat_<double>(3, 1) << -0.30, 0.00, 0.25);

    const bool ok = e3.correct(0, z, Rm);

    const double after_cx = e3.state().at<double>(CX);
    const double after_vx = e3.state().at<double>(VX);
    const double after_th = e3.state().at<double>(THETA);
    const double after_r  = e3.state().at<double>(R_ARM);
    const double after_P  = e3.cov().at<double>(CX, CX);

    printf("  correct 返回: %s\n", ok ? "true（采纳）" : "false（拒绝）");
    printf("  c_x   : %.6f -> %.6f\n", before_cx, after_cx);
    printf("  P(0,0): %.6e -> %.6e\n", before_P,  after_P);
    printf("  v_x   : %.6f -> %.6f\n", before_vx, after_vx);
    printf("  theta : %.6f -> %.6f\n", before_th, after_th);
    printf("  r     : %.6f -> %.6f\n", before_r,  after_r);

    // ======================================================================
    //  阶段 4：离谱观测应被拒绝
    // ======================================================================
    printf("\n=== 阶段 4：离谱观测 ===\n");

    cv::Mat z_bad = (cv::Mat_<double>(3, 1) << -0.30, 0.0, 20.0);
    const bool ok_bad = e3.correct(0, z_bad, Rm);
    printf("  返回: %s\n", ok_bad ? "true（✗ 不该采纳）" : "false（✓ 正确拒绝）");

    // ======================================================================
    //  阶段 5：P 的对称性
    // ======================================================================
    printf("\n=== 阶段 5：P 的对称性 ===\n");

    const cv::Mat Pm   = e3.cov();
    const double  asym = cv::norm(Pm - Pm.t(), cv::NORM_INF);
    printf("  最大不对称度: %.3e\n", asym);
    printf("  %s\n", (asym < 1e-9) ? "✓ P 对称" : "✗ P 不对称");

    // ======================================================================
    //  阶段 6：数值差分验证 H()
    // ======================================================================
    printf("\n=== 阶段 6：数值差分验证 H() ===\n");

    cv::Mat xg = cv::Mat::zeros(DIM, 1, CV_64F);
    xg.at<double>(CX)    = 0.8;
    xg.at<double>(CY)    = -0.3;
    xg.at<double>(ZA)    = 0.25;
    xg.at<double>(THETA) = 1.0;
    xg.at<double>(R_ARM) = PLATE_RADIUS;

    {
        const double eps = 1e-6;
        int    bad_count = 0;
        double max_err   = 0.0;

        for (int j = 0; j < PLATE_NUM; ++j)
        {
            const cv::Mat Hana = ekf.H(j, xg);

            for (int col = 0; col < DIM; ++col)
            {
                cv::Mat xp = xg.clone();
                cv::Mat xm = xg.clone();
                xp.at<double>(col) += eps;
                xm.at<double>(col) -= eps;

                const cv::Mat dnum = (ekf.h(j, xp) - ekf.h(j, xm)) / (2.0 * eps);

                for (int row = 0; row < 3; ++row)
                {
                    const double ana = Hana.at<double>(row, col);
                    const double num = dnum.at<double>(row);
                    const double err = std::abs(ana - num);

                    if (err > max_err) max_err = err;
                    if (err > 1e-5)
                    {
                        printf("  ✗ 板%d 状态%d 行%d : 解析=%.6f 数值=%.6f\n",
                               j, col, row, ana, num);
                        ++bad_count;
                    }
                }
            }
        }

        printf("  检查 108 个元素，最大误差: %.3e\n", max_err);
        printf("  %s\n", (bad_count == 0) ? "✓ 全部一致，H 写对了" : "✗ 有不一致");
    }

    // ======================================================================
    //  阶段 7：合成数据收敛测试
    //     第一次跑 OMEGA_TRUE = 0.0（纯平移）
    //     第二次改成 1.2（加自转）
    // ======================================================================
    printf("\n=== 阶段 7：合成数据收敛测试 ===\n");

    const double OMEGA_TRUE = 0.0;

    auto truth = [OMEGA_TRUE](double t) {
        cv::Mat s = cv::Mat::zeros(DIM, 1, CV_64F);
        s.at<double>(CX)    = 1.0 + 0.60 * t;
        s.at<double>(VX)    = 0.60;
        s.at<double>(CY)    = 0.30 + 0.20 * t;
        s.at<double>(VY)    = 0.20;
        s.at<double>(ZA)    = 0.25;
        s.at<double>(VZ)    = 0.00;
        s.at<double>(THETA) = 0.40 + OMEGA_TRUE * t;
        s.at<double>(OMEGA) = OMEGA_TRUE;
        s.at<double>(R_ARM) = PLATE_RADIUS;
        return s;
    };

    auto gauss = []() {
        const double u1 = (std::rand() + 1.0) / (RAND_MAX + 2.0);
        const double u2 = (std::rand() + 1.0) / (RAND_MAX + 2.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };

    std::srand(42);

    {
        Ekf ekf7;
        const double dt7  = 0.0415;
        const double sig7 = 0.05;
        cv::Mat R7 = cv::Mat::eye(3, 3, CV_64F) * (sig7 * sig7);

        const int N = 300;
        double sum_pos = 0.0, sum_th = 0.0;
        int    cnt = 0;

        printf("     t    真值cx  估计cx  |  真值cy  估计cy  |  真值θ  估计θ  |  ω估计\n");

        for (int k = 1; k <= N; ++k)
        {
            const double t = k * dt7;
            ekf7.predict(dt7);

            const int j = k % PLATE_NUM;
            const cv::Mat xs = truth(t);
            const cv::Mat p_true = ekf7.h(j, xs);

            cv::Mat zz = p_true.clone();
            for (int r = 0; r < 3; ++r)
                zz.at<double>(r) += sig7 * gauss();

            ekf7.correct(j, zz, R7);

            if (k > 100)
            {
                const cv::Mat xe = ekf7.state();
                const double dcx = xe.at<double>(CX) - xs.at<double>(CX);
                const double dcy = xe.at<double>(CY) - xs.at<double>(CY);
                sum_pos += std::sqrt(dcx * dcx + dcy * dcy);

                double dth = xe.at<double>(THETA) - xs.at<double>(THETA);
                while (dth >  M_PI) dth -= 2.0 * M_PI;
                while (dth < -M_PI) dth += 2.0 * M_PI;
                sum_th += std::abs(dth);
                ++cnt;
            }

            if (k % 50 == 0)
            {
                const cv::Mat xe = ekf7.state();
                printf("  %5.2f   %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f\n",
                       t, xs.at<double>(CX), xe.at<double>(CX),
                          xs.at<double>(CY), xe.at<double>(CY),
                          xs.at<double>(THETA), xe.at<double>(THETA),
                          xe.at<double>(OMEGA));
            }
        }

        printf("\n  收敛期后平均位置误差: %.4f m\n", sum_pos / cnt);
        printf("  收敛期后平均朝向误差: %.4f rad\n", sum_th  / cnt);
        printf("  %s\n", (sum_pos / cnt < 0.15) ? "✓ 位置已收敛" : "✗ 位置没收敛");
    }

    // ======================================================================
    //  阶段 9a：旋转矩阵合法性
    // ======================================================================
    printf("\n=== 阶段 9a：旋转矩阵合法性 ===\n");

    auto rotErr = [](const cv::Mat& R) -> double {
        const cv::Mat I = R * R.t();
        double e = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k)
            {
                const double want = (i == k) ? 1.0 : 0.0;
                e = std::max(e, std::abs(I.at<double>(i, k) - want));
            }
        e = std::max(e, std::abs(cv::determinant(R) - 1.0));
        return e;
    };

    for (int v = 0; v < 2; ++v)
    {
        const cv::Mat R = ImuBuffer::camToBody(v);
        printf("  camToBody(%d): R·Rᵀ误差=%.2e  det=%.6f  偏差=%.2e  %s\n",
               v,
               cv::norm(R * R.t() - cv::Mat::eye(3, 3, CV_64F), cv::NORM_INF),
               cv::determinant(R), rotErr(R),
               (rotErr(R) < 1e-9) ? "✓" : "✗");
    }

    {
        cv::Mat p = (cv::Mat_<double>(3, 1) << 0.5, -0.3, 1.45);
        for (int v = 0; v < 2; ++v)
        {
            const cv::Mat q = ImuBuffer::camToBody(v) * p;
            printf("  camToBody(%d)*p : |p|=%.6f  |q|=%.6f  %s\n",
                   v, cv::norm(p), cv::norm(q),
                   (std::abs(cv::norm(p) - cv::norm(q)) < 1e-12) ? "✓ 长度不变" : "✗");
        }
    }

    // ======================================================================
    //  阶段 9b：quatToMat 正确性
    // ======================================================================
    printf("\n=== 阶段 9b：quatToMat 正确性 ===\n");

    {
        const cv::Mat R = ImuBuffer::quatToMat(
            0.34009460359164867,
            0.023098266127927033,
            0.010465548368814045,
            0.9400492556281628);

        printf("  R =\n");
        for (int i = 0; i < 3; ++i)
            printf("    [%9.5f %9.5f %9.5f]\n",
                   R.at<double>(i, 0), R.at<double>(i, 1), R.at<double>(i, 2));

        printf("  偏差 = %.2e  det = %.6f  %s\n",
               rotErr(R), cv::determinant(R),
               (rotErr(R) < 1e-9) ? "✓ 是合法旋转矩阵" : "✗");
    }

    // ======================================================================
    //  阶段 9c：4 种坐标系约定
    // ======================================================================
    printf("\n=== 阶段 9c：坐标系约定自检 ===\n");

    ImuBuffer imu;
    if (!imu.load("data/2026-05-20_23-57-31.txt")) return 1;

    std::vector<cv::Mat> cam_pts;
    std::vector<double>  stamps;

    {
        std::ifstream f("meas.csv");
        if (!f) { printf("  ✗ 打不开 meas.csv（要在工程根目录运行）\n"); return 1; }

        std::string line;
        std::getline(f, line);                       // 跳过表头

        while (std::getline(f, line) && cam_pts.size() < 30)
        {
            int    frame = 0, okm = 0;
            double px = 0.0, py = 0.0, pz = 0.0;

            if (std::sscanf(line.c_str(), "%d,%lf,%lf,%lf,%d",
                            &frame, &px, &py, &pz, &okm) != 5) continue;
            if (okm != 1) continue;

            cam_pts.push_back((cv::Mat_<double>(3, 1) << px, py, pz));
            stamps.push_back(imu.stampOf(frame));
        }
    }

    printf("  取到 %zu 个有效量测\n\n", cam_pts.size());

    // 打印 span（跨度），用来判断哪组最"稳"
    for (int mode = 0; mode < 2; ++mode)
    {
        for (int variant = 0; variant < 2; ++variant)
        {
            double zmin = 1e9, zmax = -1e9, rmin = 1e9, rmax = -1e9;

            for (size_t i = 0; i < cam_pts.size(); ++i)
            {
                const cv::Mat pw = imu.camToWorld(cam_pts[i], stamps[i], mode, variant);
                const double z = pw.at<double>(2);
                const double r = std::sqrt(pw.at<double>(0) * pw.at<double>(0)
                                         + pw.at<double>(1) * pw.at<double>(1));
                zmin = std::min(zmin, z); zmax = std::max(zmax, z);
                rmin = std::min(rmin, r); rmax = std::max(rmax, r);
            }

            printf("  mode=%d variant=%d : z=%7.3f~%7.3f (跨度 %.3f)  |xy|=%.3f~%.3f\n",
                   mode, variant, zmin, zmax, zmax - zmin, rmin, rmax);
        }
    }
    printf("  → 选【z 跨度最小】的那一组 ✓\n");

    // ======================================================================
    //  阶段 10：世界系轨迹（前 20 帧）
    //     约定已标定为 mode=0, variant=0
    // ======================================================================
    constexpr int IMU_MODE    = 0;
    constexpr int IMU_VARIANT = 0;

    printf("\n=== 阶段 10：世界系轨迹（前 20 帧，mode=%d variant=%d）===\n",
           IMU_MODE, IMU_VARIANT);
    printf("   帧        x          y          z\n");

    {
        std::ifstream f("meas.csv");
        std::string line;
        std::getline(f, line);

        int shown = 0;
        while (std::getline(f, line) && shown < 20)
        {
            int    frame = 0, okm = 0;
            double px = 0.0, py = 0.0, pz = 0.0;

            if (std::sscanf(line.c_str(), "%d,%lf,%lf,%lf,%d",
                            &frame, &px, &py, &pz, &okm) != 5) continue;
            if (okm != 1) continue;

            const cv::Mat p_cam = (cv::Mat_<double>(3, 1) << px, py, pz);
            const cv::Mat p_w   = imu.camToWorld(p_cam, imu.stampOf(frame),
                                                 IMU_MODE, IMU_VARIANT);

            printf("  %4d   %9.3f  %9.3f  %9.3f\n", frame,
                   p_w.at<double>(0), p_w.at<double>(1), p_w.at<double>(2));
            ++shown;
        }
    }

    // ======================================================================
    //  阶段 11：离线 tracker（匹配 + 跳变处理）
    // ======================================================================
    printf("\n=== 阶段 11：离线 tracker ===\n");

    {
        ArmorTracker tracker(11.34, 0.10);

        std::ifstream fin("meas.csv");
        std::ofstream fout("state.csv");

        if (!fin) { printf("  ✗ 打不开 meas.csv\n"); return 1; }

        fout << "frame,t,cx,vx,cy,vy,za,vz,theta,omega,r,matched,"
             << "P_cx,P_cy,P_theta,raw_z\n";
        std::string line;
        std::getline(fin, line);                    // 跳过表头

        int total = 0, n_matched = 0, n_miss = 0;
        int last_matched = -1;
        int switches = 0;
        int reinit = 0;
        bool was_inited = tracker.inited();

        while (std::getline(fin, line))
        {
            int    frame = 0, okm = 0;
            double px = 0.0, py = 0.0, pz = 0.0;
            if (std::sscanf(line.c_str(), "%d,%lf,%lf,%lf,%d",
                            &frame, &px, &py, &pz, &okm) != 5) continue;

            ++total;
            const double t = imu.stampOf(frame);

            std::vector<PlateMeas> ms;
            double raw_z = 0.0;
            if (okm == 1)
            {
                const cv::Mat p_cam = (cv::Mat_<double>(3, 1) << px, py, pz);
                PlateMeas m;
                m.pos = imu.camToWorld(p_cam, t, 0, 0);
                raw_z = m.pos.at<double>(2);
                ms.push_back(m);
            }

            /* ★ 人为注入异常：第 300 帧把观测强行偏移 1.5 m
            if (frame == 300 && !ms.empty())
            {
                ms[0].pos.at<double>(0) += 1.5;
                printf("  [注入] 第 300 帧观测 x 偏移 +1.5 m\n");
            }*/

            const int j = tracker.update(ms, t);

            // 统计"重建"次数
            if (!was_inited && tracker.inited()) ++reinit;
            was_inited = tracker.inited();

            if (j >= 0)
            {
                ++n_matched;
                if (last_matched >= 0 && j != last_matched) ++switches;
                last_matched = j;
            }
            else
            {
                ++n_miss;
            }

            const cv::Mat s = tracker.ekf().state();
            const cv::Mat P = tracker.ekf().cov();

             fout << frame << "," << t << ","
                 << s.at<double>(CX)    << "," << s.at<double>(VX)    << ","
                 << s.at<double>(CY)    << "," << s.at<double>(VY)    << ","
                 << s.at<double>(ZA)    << "," << s.at<double>(VZ)    << ","
                 << s.at<double>(THETA) << "," << s.at<double>(OMEGA) << ","
                 << s.at<double>(R_ARM) << ","
                 << j << ","
                 << P.at<double>(CX, CX)       << ","
                 << P.at<double>(CY, CY)       << ","
                 << P.at<double>(THETA, THETA) << ","
                 << raw_z << "\n";
        }

        printf("  总帧数       : %d\n", total);
        printf("  匹配成功     : %d  (%.1f%%)\n", n_matched,
                100.0 * n_matched / std::max(total, 1));
        printf("  未匹配       : %d\n", n_miss);
        printf("  板号切换次数 : %d\n", switches);
        printf("  重建次数     : %d\n", reinit);
        printf("  → state.csv 已导出\n");
    }

    printf("\n=== 全部测试结束 ===\n");
    return 0;
}