#pragma once

#include "opencv2/stitching/detail/matchers.hpp"
#include <vector>


#include "defs.h"

extern void custom_resize(cv::cuda::GpuMat &in, cv::cuda::GpuMat &out, cv::Size t_size);
namespace meshwarp {
    void calibrateMeshWarp(std::vector<cv::Mat> &full_imgs, std::vector<cv::detail::ImageFeatures> &features,
                           std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                           std::vector<cv::cuda::GpuMat> &x_mesh, std::vector<cv::cuda::GpuMat> &y_mesh,
                           std::vector<cv::cuda::GpuMat> &x_maps, std::vector<cv::cuda::GpuMat> &y_maps,
                           float focal_length, double compose_scale, double work_scale);

    void recalibrateMesh(std::vector<cv::Mat> &full_img, std::vector<cv::cuda::GpuMat> &x_maps,
                         std::vector<cv::cuda::GpuMat> &y_maps, std::vector<cv::cuda::GpuMat> &x_mesh,
                         std::vector<cv::cuda::GpuMat> &y_mesh, float focal_length, double compose_scale,
                         const double &work_scale);
}
