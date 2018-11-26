#pragma once

#include "opencv2/cudaarithm.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include <vector>

#include "defs.h"

extern void custom_resize(cv::cuda::GpuMat &in, cv::cuda::GpuMat &out, cv::Size t_size);

void findFeatures(std::vector<cv::Mat> &full_img, std::vector<cv::detail::ImageFeatures> &features,
                  const double &work_scale);

void matchFeatures(std::vector<cv::detail::ImageFeatures> &features, std::vector<cv::detail::MatchesInfo> &pairwise_matches);

bool calibrateCameras(std::vector<cv::detail::CameraParams> &cameras,
                      const cv::Size full_img_size, const double work_scale);


void warpImages(std::vector<cv::Mat> full_img, cv::Size full_img_size,
                std::vector<cv::detail::CameraParams> cameras, cv::Ptr<cv::detail::Blender> blender,
                cv::Ptr<cv::detail::ExposureCompensator> compensator, double work_scale,
                double seam_scale, double seam_work_aspect, std::vector<cv::cuda::GpuMat> &x_maps,
                std::vector<cv::cuda::GpuMat> &y_maps, double &compose_scale,
                float &warped_image_scale, float &blend_width);


void calibrateMeshWarp(std::vector<cv::Mat> &full_imgs, std::vector<cv::detail::ImageFeatures> &features,
                       std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                       std::vector<cv::cuda::GpuMat> &x_mesh, std::vector<cv::cuda::GpuMat> &y_mesh,
                       std::vector<cv::cuda::GpuMat> &x_maps, std::vector<cv::cuda::GpuMat> &y_maps,
                       float focal_length, double compose_scale, double work_scale);


bool stitch_calib(std::vector<cv::Mat> full_img, std::vector<cv::detail::CameraParams> &cameras,
                  std::vector<cv::cuda::GpuMat> &x_maps, std::vector<cv::cuda::GpuMat> &y_maps,
                  std::vector<cv::cuda::GpuMat> &x_mesh, std::vector<cv::cuda::GpuMat> &y_mesh,
                  double &work_scale, double &seam_scale, double &seam_work_aspect, double &compose_scale,
                  cv::Ptr<cv::detail::Blender> &blender, cv::Ptr<cv::detail::ExposureCompensator> compensator,
                  float &warped_image_scale, float &blend_width, cv::Size &full_img_size);

void recalibrateMesh(std::vector<cv::Mat> &full_img, std::vector<cv::cuda::GpuMat> &x_maps,
                     std::vector<cv::cuda::GpuMat> &y_maps, std::vector<cv::cuda::GpuMat> &x_mesh,
                     std::vector<cv::cuda::GpuMat> &y_mesh, float focal_length, double compose_scale,
	                 const double &work_scale);
