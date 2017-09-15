project(autodiff)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
AUX_SOURCE_DIRECTORY(. DIR_SRCS)

#add_subdirectory(CL_files)
#add_subdirectory(src)

#################指定OpenCL库依赖, 手动指定方式#######################
#指定Opencl的目录
SET(OPENCL_INCLUDE_PATH /usr/include CACHE PATH "Path to OpenCL include files")
SET(OPENCL_LIB_PATH /usr/lib CACHE PATH "Path to OpenCL libraries")
include_directories(${OPENCL_INCLUDE_PATH})  #指定OpenCL头文件的搜索目录
link_directories(${OPENCL_LIB_PATH})  #指定OpenCL库的搜索目录

################指定OpenCL库依赖, find_package方式#######################
#find_package(opencl)
#include_directories(${OpenCL_INCLUDE_DIR})
#LINK_DIRECTORIES(${OpenCL_LIBRARY})
#TARGET_LINK_LIBRARIES(PSBA OpenCL) #实际会形成OpenCL.lib库依赖


#放在最后
ADD_EXECUTABLE(autodiff ${DIR_SRCS}) #最终生成的可执行命令
TARGET_LINK_LIBRARIES(autodiff OpenCL)  #实际会形成OpenCL.lib库依赖
