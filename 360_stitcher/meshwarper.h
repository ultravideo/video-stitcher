#pragma once

#include "opencv2/stitching/detail/matchers.hpp"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"
#include <vector>
#include <mutex>


#include "defs.h"

extern std::mutex mesh_mutex;

class MeshWarper {
public:
    MeshWarper(int num_images, int M, int N, float focal_length, double compose_scale, double work_scale);
    void calibrateMeshWarp(std::vector<cv::Mat> &full_imgs, std::vector<cv::detail::ImageFeatures> &features,
                           std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                           std::vector<cv::cuda::GpuMat> &x_mesh, std::vector<cv::cuda::GpuMat> &y_mesh,
                           std::vector<cv::cuda::GpuMat> &x_maps, std::vector<cv::cuda::GpuMat> &y_maps);

    void createMesh(std::vector<cv::Mat> &full_imgs, std::vector<cv::detail::ImageFeatures> &features,
                    std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                    std::vector<cv::Mat> &x_mesh, std::vector<cv::Mat> &y_mesh,
                    std::vector<cv::cuda::GpuMat> &x_maps, std::vector<cv::cuda::GpuMat> &y_maps,
                    std::vector<cv::Size> &mesh_size);

    void recalibrateMesh(std::vector<cv::Mat> &full_img, std::vector<cv::cuda::GpuMat> &x_maps,
                         std::vector<cv::cuda::GpuMat> &y_maps, std::vector<cv::cuda::GpuMat> &x_mesh,
                         std::vector<cv::cuda::GpuMat> &y_mesh);

    std::vector<cv::Mat> interpolateMesh(std::vector<cv::Mat> &meshes_start, std::vector<cv::Mat> &meshes_end, float progress);

    void convertMeshToMap(cv::Mat &mesh_cpu_x, cv::Mat &mesh_cpu_y,
                          cv::cuda::GpuMat &map_x, cv::cuda::GpuMat &map_y, cv::Size mesh_size);


private:
    // dst is the image id of the matches train image
    typedef struct matchWithDst {
        cv::DMatch match;
        int dst;
    } matchWithDst_t;

    // @row means number of rows already used inside SparseMatrix A and VectorXd b
    // calcTerm methods update @row automatically.
    void calcGlobalTerm(int &row, const std::vector<cv::Point> &features, cv::Size mesh_size, int idx);
    void calcSmoothnessTerm(int &row, const cv::Mat &image, cv::Size mesh_size, int idx);
    void calcLocalTerm(int &row, const std::vector<matchWithDst_t> &matches,
                       const std::vector<cv::detail::ImageFeatures> &features, int idx);

    cv::Mat drawMesh(const cv::Mat &mesh_cpu_x, const cv::Mat &mesh_cpu_y, cv::Size mesh_size);
    void convertVectorToMesh(const Eigen::VectorXd &x, cv::Mat &out_mesh_x, cv::Mat &out_mesh_y, int idx);


    Eigen::SparseMatrix<double> A; // coefficients
    Eigen::VectorXd b; // constant coefficients
    int M; // number of vertexes in the mesh horizontally
    int N; // number of vertexes in the mesh vertically
    float focal_length;
    double compose_scale;
    double work_scale;
};
