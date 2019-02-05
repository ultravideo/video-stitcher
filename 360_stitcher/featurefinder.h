#pragma once
#include <vector>

#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"

namespace featurefinder {
    void findFeatures(std::vector<cv::Mat> &full_img, std::vector<cv::Mat> &masks, std::vector<cv::detail::ImageFeatures> &features,
                      const double &work_scale);

    void matchFeatures(std::vector<cv::detail::ImageFeatures> &features, std::vector<cv::detail::MatchesInfo> &pairwise_matches);

}
