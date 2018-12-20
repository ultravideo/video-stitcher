#pragma once

#include "opencv2/stitching/detail/matchers.hpp"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"
#include <vector>


#include "defs.h"

extern void custom_resize(cv::cuda::GpuMat &in, cv::cuda::GpuMat &out, cv::Size t_size);
namespace meshwarp {
    // dst is the image id of matches train image
    typedef struct matchWithDst {
        cv::DMatch match;
        int dst;
    } matchWithDst_t;

    void calibrateMeshWarp(std::vector<cv::Mat> &full_imgs, std::vector<cv::detail::ImageFeatures> &features,
                           std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                           std::vector<cv::cuda::GpuMat> &x_mesh, std::vector<cv::cuda::GpuMat> &y_mesh,
                           std::vector<cv::cuda::GpuMat> &x_maps, std::vector<cv::cuda::GpuMat> &y_maps,
                           float focal_length, double compose_scale, double work_scale);

    void recalibrateMesh(std::vector<cv::Mat> &full_img, std::vector<cv::cuda::GpuMat> &x_maps,
                         std::vector<cv::cuda::GpuMat> &y_maps, std::vector<cv::cuda::GpuMat> &x_mesh,
                         std::vector<cv::cuda::GpuMat> &y_mesh, float focal_length, double compose_scale,
                         const double &work_scale);

    // @start means the the term will start at the index @start
    void calcGlobalTerm(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &b, int start);
    void calcSmoothnessTerm(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &b, int start);
    void calcLocalTerm(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &b, int start);


    cv::Mat drawMesh(const cv::Mat &mesh_cpu_x, const cv::Mat &mesh_cpu_y, cv::Size mesh_size);
    //TODO: Don't use global N and M
    void convertVectorToMesh(const Eigen::VectorXd &x, cv::Mat &out_mesh_x, cv::Mat &out_mesh_y, int idx);
    void convertMeshToMap(cv::Mat &mesh_cpu_x, cv::Mat &mesh_cpu_y, cv::cuda::GpuMat &map_x, cv::cuda::GpuMat &map_y, cv::Size mesh_size);
}
