#pragma once

#include "defs.h"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaarithm.hpp"

bool showMat(const char *name, const cv::Mat &m);
bool showMat(const char *name, const cv::UMat &m);
bool showMat(const char *name, const cv::cuda::GpuMat &m);

void showMats(const char *name, const std::vector<cv::Mat> &mats);
void showMats(const char *name, const std::vector<cv::UMat> &mats);
void showMats(const char *name, const std::vector<cv::cuda::GpuMat> &mats);
