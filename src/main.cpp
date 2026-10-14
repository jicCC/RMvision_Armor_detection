#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yolo_detector.hpp"
#include "pnp_solver.hpp"

int main()
{
    YoloDetector detector("data/armor.onnx");
    PnpSolver    solver(869.1168824543144, 869.1168824543144, 480.0, 360.0);
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

        std::vector<Detection> dets = detector.detect(img);      // 先识别

        for (const auto& d : dets) 
        {                             // 打印，用于验证
            std::cout << "cls=" << d.cls << "  score=" << d.score << "\n";
            for (int k = 0; k < 4; ++k)
                std::cout << "  kpt" << k << " = (" << d.kpt[k].x << ", " << d.kpt[k].y << ")\n";

            std::vector<cv::Point2f> img_ordered = { d.kpt[0], d.kpt[3], d.kpt[2], d.kpt[1] };

            PnpResult solve_result = solver.solve(img_ordered);
            if (!solve_result.ok) {
                std::cout << "   解算失败" << std::endl;
                continue;
            }
            std::cout << "   cls=" << d.cls << " score=" << d.score
                      << "  x=" << solve_result.tvec.at<double>(0)
                      << "  y=" << solve_result.tvec.at<double>(1)
                      << "  z=" << solve_result.tvec.at<double>(2)
                      << "  距离=" << solve_result.distance << " m"
                      << "  误差=" << solve_result.reproj_error << std::endl;
        }

        detector.draw(img, dets);  
        std::string save_name = "result_" + std::to_string(i + 1) + ".jpg";
        cv::imwrite(save_name, img);
        std::cout << "已保存: " << save_name << std::endl;

        
        
        
    }

    return 0;
}