#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yolo_detector.hpp"
#include "pnp_solver.hpp"
#include "traditional_detector.hpp"
#include <cstdio>


// ============================================================
//  对一组检测结果做 PnP 解算并打印
//  两种方案共用：因为它们都返回 std::vector<Detection>，
//  且四个角点都是 canonical 顺序（左上→右上→右下→左下）
// ============================================================
static void solveAndPrint(const std::vector<Detection>& dets,
                          PnpSolver& solver,
                          const std::string& tag)
{
    for (const auto& d : dets)
    {
        // 解算：输入四个角点，输出位姿
        PnpResult r = solver.solve(d.kpt);

        std::cout << tag << "  cls=" << d.cls << "  score=" << d.score;

        // 解算失败就不用往下取 tvec 了（里面是空的）
        if (!r.ok)
        {
            std::cout << "  [解算失败]" << std::endl;
            continue;
        }

        std::cout << "  x=" << r.tvec.at<double>(0, 0)
                  << "  y=" << r.tvec.at<double>(1, 0)
                  << "  z=" << r.tvec.at<double>(2, 0)
                  << "  距离=" << r.distance << " m"
                  << "  误差=" << r.reproj_error << std::endl;
    }
}



struct GtPlate //真值表
{
    const char* name;      // 板编号
    double      x, y, z;   // 真值位置（m）
};

static const std::vector<GtPlate> GT_OF_IMAGE[3] = {
    // test1.jpg → armor_1、armor_4 正面
    { {"armor_1", -0.174340, 0.169600, 3.794632},
      {"armor_4",  0.119301, 0.183937, 3.762585} },
    // test2.jpg → armor_1、armor_2 正面
    { {"armor_1",  0.285747, 0.070019, 1.384573},
      {"armor_2",  0.009916, 0.097664, 1.281592} },
    // test3.jpg → armor_3、armor_4 正面
    { {"armor_3",  0.189233, 0.070265, 1.553395},
      {"armor_4", -0.092414, 0.092079, 1.467008} },
};




//  findGtMatch —— 给定一次解算结果，找出它对应哪块真值板
static int findGtMatch(const cv::Mat& tvec, const std::vector<GtPlate>& gt)
{
    int    best   = -1;      // 目前最匹配的真值板下标
    double best_d = 1e9;     // 目前最小的三维距离（初值给一个很大的数）

    for (size_t i = 0; i < gt.size(); ++i)
    {
        // 计算 tvec 与这块真值板的三维差
        double dx = tvec.at<double>(0, 0) - gt[i].x;
        double dy = tvec.at<double>(1, 0) - gt[i].y;
        double dz = tvec.at<double>(2, 0) - gt[i].z;

        // 三维距离
        double d = std::sqrt(dx * dx + dy * dy + dz * dz);

        // 比当前最近还近，就替换
        if (d < best_d)
        {
            best_d = d;
            best   = static_cast<int>(i);
        }
    }
    return best;
}



//  printErrorTable —— 打印一种方案在这张图上的误差表
static void printErrorTable(const std::vector<Detection>& dets,
                            PnpSolver& solver,
                            const std::vector<GtPlate>& gt,
                            const std::string& tag)
{
    std::printf("  [方案] | [真值板]  |          Δx / Δy / Δz (m)          | 三维误差(m) |  Δz%%\n");

    for (const auto& d : dets)
    {
        //解算
        PnpResult r = solver.solve(d.kpt);
        if (!r.ok)
        {
            std::printf("  %s | [解算失败]\n", tag.c_str());
            continue;
        }

        //找到对应的真值板
        int gi = findGtMatch(r.tvec, gt);
        if (gi < 0) continue;

        //算误差（Δ = 计算值 − 真值）
        double dx = r.tvec.at<double>(0, 0) - gt[gi].x;
        double dy = r.tvec.at<double>(1, 0) - gt[gi].y;
        double dz = r.tvec.at<double>(2, 0) - gt[gi].z;

        double e3 = std::sqrt(dx * dx + dy * dy + dz * dz);

        //打印一行
        std::printf("  %s | %-8s | dx=%+8.4f  dy=%+8.4f  dz=%+8.4f | %8.4f    | %+6.1f%%\n",
                    tag.c_str(),
                    gt[gi].name,
                    dx, dy, dz,
                    e3,
                    100.0 * dz / gt[gi].z);
    }
}



int main()
{
    YoloDetector detector("data/armor.onnx");
    PnpSolver    solver(869.1168824543144, 869.1168824543144, 480.0, 360.0);
    TraditionalDetector trad(true, "data/mlp.onnx");       // true = 蓝方
    std::vector<std::string> paths = 
    {
        "data/test1.jpg",
        "data/test2.jpg",
        "data/test3.jpg"
    };

    
    for (int i = 0; i < paths.size(); i++) 
    {
        cv::Mat img = cv::imread(paths[i]);
        if (img.empty())
        {
            std::cout << "读不到图: " << paths[i] << std::endl;
            continue;
        } 


        // ================= 传统视觉方案 =================
        std::vector<Detection> trad_dets = trad.detect(img);
        solveAndPrint(trad_dets, solver, "[传统]");

        // 单独一张图，只画传统方案的黄框
        cv::Mat img_trad = img.clone();
        trad.draw(img_trad, trad_dets);

        // ================= 神经网络方案 =================
        std::vector<Detection> yolo_dets = detector.detect(img);
        solveAndPrint(yolo_dets, solver, "[神经]");


        // ---------- 误差表（与真值对比）----------
            std::printf("\n[误差表] 第 %d 张图\n", int(i) + 1);
            printErrorTable(trad_dets, solver, GT_OF_IMAGE[i], "传统");
            printErrorTable(yolo_dets, solver, GT_OF_IMAGE[i], "神经");

            
        // 单独一张图，只画神经网络的绿框
        cv::Mat img_yolo = img.clone();
        detector.draw(img_yolo, yolo_dets);

        // ================= 叠加图（报告里的"角点对比"插图用）=================
        cv::Mat img_cmp = img.clone();
        trad.draw(img_cmp, trad_dets);        // 先画粗的黄色
        detector.draw(img_cmp, yolo_dets);    // 再画细的绿色，压在黄线上面

        // ================= 保存三个文件 =================
        // 从原来的文件名派生：result_1.jpg → result_1_trad.jpg / _yolo.jpg / _cmp.jpg
        std::string img_file = "result_" + std::to_string(i) + ".jpg";
        std::string base = img_file.substr(0, img_file.find_last_of('.'));   // "result_1"
        cv::imwrite(base + "_trad.jpg", img_trad);
        cv::imwrite(base + "_yolo.jpg", img_yolo);
        cv::imwrite(base + "_cmp.jpg",  img_cmp);
       
    }

    return 0;
}