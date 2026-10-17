cmake_minimum_required(VERSION 3.10)
project(rm_vision_hw)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_executable(test_opencv src/main.cpp   src/yolo_detector.cpp   src/pnp_solver.cpp  
               src/traditional_detector.cpp)

target_link_libraries(test_opencv ${OpenCV_LIBS})
target_link_libraries(test_opencv openvino::runtime)