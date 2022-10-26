set(CMAKE_PREFIX_PATH /home/wj/ai/software/libtorch)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_VERBOSE_MAKEFILE on)
add_definitions(-DUSE_TORCH)
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(faster_rcnn)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
aux_source_directory(. SRC_FILES)

# Define our library target
#add_library(faster_rcnn SHARED $SRC_FILES)
link_directories(/home/wj/ai/work/torchvision_ops/build)
add_executable(faster_rcnn ${SRC_FILES})

include_directories(/usr/local/include/opencv4/opencv2)
include_directories(/usr/local/include/opencv4)
# Enable C++14
target_compile_features(faster_rcnn PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(faster_rcnn "${TORCH_LIBRARIES}")
# Link against OpenCV
#target_link_libraries(faster_rcnn opencv_core opencv_imgproc)
target_link_libraries(faster_rcnn ${OpenCV_LIBS})
target_link_libraries(faster_rcnn visionop)
#target_link_libraries(faster_rcnn /home/wj/ai/software/vision_ops/build/libvisionop.a)
target_link_options(faster_rcnn PUBLIC -Wl,-rpath,/home/wj/ai/work/torchvision_ops/build)
