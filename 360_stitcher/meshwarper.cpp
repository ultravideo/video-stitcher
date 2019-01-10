#include "meshwarper.h"
#include "featurefinder.h"
#include "debug.h"

#include "opencv2/imgproc.hpp"




using namespace cv;
using namespace cv::detail;
using std::vector;
extern void custom_resize(cv::cuda::GpuMat &in, cv::cuda::GpuMat &out, cv::Size t_size);


MeshWarper::MeshWarper(int num_images, int mesh_width, int mesh_height, 
                       float focal_length, double compose_scale, double work_scale):
    M(mesh_width), N(mesh_height), focal_length(focal_length),
    compose_scale(compose_scale), work_scale(work_scale)
{
    int features_per_image = MAX_FEATURES_PER_IMAGE;
    // 2 dimensions, x and y
    int dims = 2;

    int global_size = dims * num_images * N * M;
    // N - 1 and M - 1, since edges will only require half compared to the middle
    int smoothness_size = dims * num_images * (N - 1) * (M - 1) * 8;
    int local_size = dims * (num_images + (int)wrapAround + 1) * features_per_image;

    // Number of rows/equations needed for calculating the mesh
    int num_rows = global_size + smoothness_size + local_size;
    A.resize(num_rows, dims * N * M * num_images);
    b.resize(num_rows);
    b.fill(0);
}

void MeshWarper::calibrateMeshWarp(vector<Mat> &full_imgs, vector<ImageFeatures> &features,
                       vector<MatchesInfo> &pairwise_matches, vector<cuda::GpuMat> &x_mesh,
                       vector<cuda::GpuMat> &y_mesh, vector<cuda::GpuMat> &x_maps,
                       vector<cuda::GpuMat> &y_maps) {
    vector<Size> mesh_size(full_imgs.size());
    vector<Mat> images(full_imgs.size());
    vector<Mat> mesh_cpu_x(full_imgs.size());
    vector<Mat> mesh_cpu_y(full_imgs.size());
    vector<Mat> x_map(full_imgs.size());
    vector<Mat> y_map(full_imgs.size());

    for (int idx = 0; idx < full_imgs.size(); ++idx) {
        mesh_cpu_x[idx] = Mat(N, M, CV_32FC1);
        mesh_cpu_y[idx] = Mat(N, M, CV_32FC1);
        resize(full_imgs[idx], images[idx], Size(), compose_scale, compose_scale);
        x_maps[idx].download(x_map[idx]);
        y_maps[idx].download(y_map[idx]);
        remap(images[idx], images[idx], x_map[idx], y_map[idx], INTER_LINEAR);
        mesh_size[idx] = images[idx].size();
        x_mesh[idx] = cuda::GpuMat(mesh_size[idx], CV_32FC1);
        y_mesh[idx] = cuda::GpuMat(mesh_size[idx], CV_32FC1);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                mesh_cpu_x[idx].at<float>(i, j) = static_cast<float>(j) * mesh_size[idx].width / (M - 1);
                mesh_cpu_y[idx].at<float>(i, j) = static_cast<float>(i) * mesh_size[idx].height / (N - 1);
            }
        }
    }

    featurefinder::findFeatures(images, features, -1);
    featurefinder::matchFeatures(features, pairwise_matches);

    int features_per_image = MAX_FEATURES_PER_IMAGE;


    vector<matchWithDst_t> all_matches[NUM_IMAGES];
    // Select all matches that fit criteria
    for (int idx = 0; idx < pairwise_matches.size(); ++idx) {
        MatchesInfo &pw_matches = pairwise_matches[idx];
        int src = pw_matches.src_img_idx;
        int dst = pw_matches.dst_img_idx;
        if (!pw_matches.matches.size() || !pw_matches.num_inliers) continue;

        if (dst != NUM_IMAGES - 1 || src != 0 && dst == 5) {
            // Only calculate loss from pairs of src and dst where src = dst - 1
            // to avoid using same pairs multiple times
            if (abs(src - dst - 1) > 0.1) {
                continue;
            }
        }
        if (VISUALIZE_MATCHES) {
            Mat visual_matches;
            drawMatches(images[src], features[src].keypoints, images[dst], features[dst].keypoints, pw_matches.matches, visual_matches);
            showMat("visualized matches", visual_matches);
        }

        //Find all indexes of the inliers_mask that contain the value 1
        for (int i = 0; i < pw_matches.inliers_mask.size(); ++i) {
            if (pw_matches.inliers_mask[i]) {
                matchWithDst_t match = { pw_matches.matches[i], pw_matches.dst_img_idx };
                all_matches[src].push_back(match);
            }
        }
    }


    vector<matchWithDst_t> selected_matches[NUM_IMAGES];

    // Select features_per_image amount of random features points from all_matches
    for (int img = 0; img < NUM_IMAGES; ++img) {
        std::random_shuffle(all_matches[img].begin(), all_matches[img].end());
        for (int i = 0; i < min(features_per_image, (int)(all_matches[img].size() * 0.8f)); ++i) {
            matchWithDst_t match = all_matches[img].at(i);
            selected_matches[img].push_back(match);
        }
    }

    vector<Point> selected_points[NUM_IMAGES];

    // Convert matchWithDst_t to points, which is needed for the global term
    for (int idx = 0; idx < images.size(); ++idx) {
        for (int i = 0; i < selected_matches[idx].size(); ++i) {
            int ft_id = selected_matches[idx].at(i).match.queryIdx;
            Point p = features[idx].keypoints.at(ft_id).pt;
            selected_points[idx].push_back(p);
        }
    }


    int rows_used = 0;
    for (int idx = 0; idx < images.size(); ++idx) {
        calcGlobalTerm(rows_used, selected_points[idx], mesh_size.at(idx), idx);
        calcSmoothnessTerm(rows_used, images.at(idx), mesh_size.at(idx), idx);
        calcLocalTerm(rows_used, selected_matches[idx], features, idx);
    }

    // Draw the meshes for visualisation
    if (VISUALIZE_MATCHES) {
        for (int idx = 0; idx < NUM_IMAGES; ++idx) {
            for (int i = 0; i < mesh_cpu_x[idx].rows - 1; ++i) {
                for (int j = 0; j < mesh_cpu_x[idx].cols; ++j) {
                    Point start = Point(mesh_cpu_x[idx].at<float>(i, j), mesh_cpu_y[idx].at<float>(i, j));
                    Point end = Point(mesh_cpu_x[idx].at<float>(i + 1, j), mesh_cpu_y[idx].at<float>(i + 1, j));
                    line(images[idx], start, end, Scalar(255, 0, 0), 1);
                }
            }
            for (int i = 0; i < mesh_cpu_x[idx].rows; ++i) {
                for (int j = 0; j < mesh_cpu_x[idx].cols - 1; ++j) {
                    Point start = Point(mesh_cpu_x[idx].at<float>(i, j), mesh_cpu_y[idx].at<float>(i, j));
                    Point end = Point(mesh_cpu_x[idx].at<float>(i, j + 1), mesh_cpu_y[idx].at<float>(i, j + 1));
                    line(images[idx], start, end, Scalar(255, 0, 0), 1);
                }
            }
        }
    }

    // LeastSquaresConjudateGradientSolver solves equations that are in format |Ax + b|^2
    Eigen::VectorXd x;
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    x = solver.solve(b);

    for (int idx = 0; idx < full_imgs.size(); ++idx) {
        convertVectorToMesh(x, mesh_cpu_x[idx], mesh_cpu_y[idx], idx);
        convertMeshToMap(mesh_cpu_x[idx], mesh_cpu_y[idx], x_mesh[idx], y_mesh[idx], mesh_size[idx]);
        if (VISUALIZE_WARPED) {
            Mat visualized_mesh = drawMesh(mesh_cpu_x[idx], mesh_cpu_y[idx], mesh_size[idx]);
            imshow(std::to_string(idx), visualized_mesh);
            waitKey();
        }
    }
}

void MeshWarper::recalibrateMesh(std::vector<cv::Mat> &full_img, std::vector<cv::cuda::GpuMat> &x_maps,
                     std::vector<cv::cuda::GpuMat> &y_maps, std::vector<cv::cuda::GpuMat> &x_mesh,
                     std::vector<cv::cuda::GpuMat> &y_mesh)
{
    vector<ImageFeatures> features(NUM_IMAGES);
    vector<MatchesInfo> pairwise_matches;
    A.setZero();
    b.fill(0);

    calibrateMeshWarp(full_img, features, pairwise_matches, x_mesh, y_mesh, x_maps, y_maps);
}

// Global alignment term from http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
void MeshWarper::calcGlobalTerm(int &row, const vector<Point> &features,
                              Size mesh_size, int idx)
{
    int img_matrix_size = N * M * 2;
    int col = img_matrix_size * idx;

    // Square root ALPHAS, because equation is in format |Ax + b|^2 instead of Ax + b
    // |sqrt(ALPHA) * (Ax + b)|^2 == ALPHA * |Ax + b|^2
    float a = sqrt(ALPHAS[1]);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            float x1 = static_cast<float>(j * mesh_size.width / (M-1));
            float y1 = static_cast<float>(i * mesh_size.height / (N-1));
            float tau = 1;
            for (int ft = 0; ft < features.size(); ++ft) {
                Point ft_point = features[ft];
                if (sqrt(pow(ft_point.x - x1, 2) + pow(ft_point.y - y1, 2)) < GLOBAL_DIST) {
                    tau = 0;
                    break;
                }
            }
            A.insert(row, col) = a * tau;
            A.insert(row + 1, col + 1) = a * tau;
            b(row) = a * tau * x1;
            b(row + 1) = a * tau * y1;
            row += 2;
            col += 2;
        }
    }
}

// Smoothness term from http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
void MeshWarper::calcSmoothnessTerm(int &row, const cv::Mat &image,
                                  Size mesh_size, int idx)
{
    // Size of an image's smoothness term
    int img_matrix_size = (N - 1) * (M - 1) * 2 * 8;
    int col = img_matrix_size * idx;

    float a = sqrt(ALPHAS[2]);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            // Loop through all triangles surrounding vertex (j, i), when the surrounding quads are split in the middle
            // Left to right, top to bottom order
            //
            //  ___________
            // |\  1 |2   /|
            // | 0\  |  /3 |
            // |____\|/____|
            // | 4  /|\  7 |
            // |  / 5| 6\  |
            // |/____|____\|
            for (int t = 0; t < 8; ++t) {

                //Indexes of the triangle vertex positions, (0,0) being (j,i) in (x, y) format
                Point2i Vi[3];

                switch (t) {
                case 0:
                    Vi[0] = {.x = -1, .y = 0};
                    Vi[1] = {.x = 0, .y = 0};
                    Vi[2] = {.x = -1, .y = -1};
                    break;
                case 1:
                    Vi[0] = {.x = 0, .y = -1};
                    Vi[1] = {.x = 0, .y = 0};
                    Vi[2] = {.x = -1, .y = -1};
                    break;
                case 2:
                    Vi[0] = {.x = 0, .y = -1};
                    Vi[1] = {.x = 0, .y = 0};
                    Vi[2] = {.x = 1, .y = -1};
                    break;
                case 3:
                    Vi[0] = {.x = 1, .y = 0};
                    Vi[1] = {.x = 0, .y = 0};
                    Vi[2] = {.x = 1, .y = -1};
                    break;
                case 4:
                    Vi[0] = {.x = -1, .y = 0};
                    Vi[1] = {.x = 0, .y = 0};
                    Vi[2] = {.x = -1, .y = 1};
                    break;
                case 5:
                    Vi[0] = {.x = 0, .y = 1};
                    Vi[1] = {.x = 0, .y = 0};
                    Vi[2] = {.x = -1, .y = 1};
                    break;
                case 6:
                    Vi[0] = {.x = 0, .y = 1};
                    Vi[1] = {.x = 0, .y = 0};
                    Vi[2] = {.x = 1, .y = 1};
                    break;
                case 7:
                    Vi[0] = {.x = 1, .y = 0};
                    Vi[1] = {.x = 0, .y = 0};
                    Vi[2] = {.x = 1, .y = 1};
                    break;
                default:
                    break;
                }

                Point2i vertex_offset = {.x = j, .y = i};

                Point2i Vi_total[3];
                Vi_total[0] = vertex_offset + Vi[0];
                Vi_total[1] = vertex_offset + Vi[1];
                Vi_total[2] = vertex_offset + Vi[2];

                bool out_of_bounds = false;
                for (int v = 0; v < 3; ++v) {
                    if (Vi_total[v].x < 0 || Vi_total[v].y < 0 || Vi_total[v].x >= M || Vi_total[v].y >= N) {
                        out_of_bounds = true;
                        break;
                    }
                }
                if (out_of_bounds)
                    continue;


                float width = mesh_size.width;
                float height = mesh_size.height;
                float V1x = Vi_total[0].x * (width / (M - 1));
                float V2x = Vi_total[1].x * (width / (M - 1));
                float V3x = Vi_total[2].x * (width / (M - 1));

                float V1y = Vi_total[0].y * (height / (N - 1));
                float V2y = Vi_total[1].y * (height / (N - 1));
                float V3y = Vi_total[2].y * (height / (N - 1));


                float x1 = V1x;
                float x2 = V2x;
                float x3 = V3x;
                float y1 = V1y;
                float y2 = V2y;
                float y3 = V3y;

                // Calculated from equation [6] http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
                // Vn is in code [xn, yn]
                float u = (-x1*y2+x1*y3-x2*y1+2*x2*y2-x2*y3+x3*y1-x3*y2)/(2*(x2-x3)*(y2-y3));
                float v = (x1*y2-x1*y3-x2*y1+x2*y3+x3*y1-x3*y2)/(2*(x2-x3)*(y2-y3));


                float sal; // Salience of the triangle. Using 0.5f + l2-norm of color values of the triangle
                Mat mask(mesh_size.height, mesh_size.width, CV_8UC1);
                mask.setTo(Scalar::all(0));
                Point pts[3] = { Point(V1x, V1y), Point(V2x, V2y), Point(V3x, V3y) };
                fillConvexPoly(mask, pts, 3, Scalar(255));

                Mat mean;
                Mat deviation;

                // TODO: meanstdDev is a very slow operation. Make it faster
                meanStdDev(image, mean, deviation, mask);
                Mat variance;
                pow(deviation, 2, variance);
                sal = sqrt(norm(variance, NORM_L2) + 0.5f);



                A.insert(row,   2*((j + Vi[0].x) + M * (i + Vi[0].y) + M*N*idx)) = a * sal; // x1
                A.insert(row,   2*((j + Vi[0].x) + M * (i + Vi[0].y) + M*N*idx) + 1) = a * sal; // y1
                A.insert(row,   2*((j + Vi[1].x) + M * (i + Vi[1].y) + M*N*idx)) = a*(u - v - 1) * sal; // x2
                A.insert(row,   2*((j + Vi[1].x) + M * (i + Vi[1].y) + M*N*idx) + 1) = a*(u + v - 1) * sal; // y2
                A.insert(row,   2*((j + Vi[2].x) + M * (i + Vi[2].y) + M*N*idx)) = a*(-u + v) * sal; // x3
                A.insert(row,   2*((j + Vi[2].x) + M * (i + Vi[2].y) + M*N*idx) + 1) = a*(-u - v) * sal; // y3

                // b should be zero anyway, but for posterity's sake set it to zero
                b(row) = 0;


                A.insert(row + 1,   2*((j + Vi[0].x) + M * (i + Vi[0].y) + M*N*idx)) = a * sal; // x1
                A.insert(row + 1,   2*((j + Vi[0].x) + M * (i + Vi[0].y) + M*N*idx) + 1) = a * sal; // y1
                A.insert(row + 1,   2*((j + Vi[1].x) + M * (i + Vi[1].y) + M*N*idx)) = a*(u - v - 1) * sal; // x2
                A.insert(row + 1,   2*((j + Vi[1].x) + M * (i + Vi[1].y) + M*N*idx) + 1) = a*(u + v - 1) * sal; // y2
                A.insert(row + 1,   2*((j + Vi[2].x) + M * (i + Vi[2].y) + M*N*idx)) = a*(-u + v) * sal; // x3
                A.insert(row + 1,   2*((j + Vi[2].x) + M * (i + Vi[2].y) + M*N*idx) + 1) = a*(-u - v) * sal; // y3

                // b should be zero anyway, but for posterity's sake set it to zero
                b(row + 1) = 0;
                row += 2;
                col += 2;
            }
        }
    }
}

// Local alignment term from http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
void MeshWarper::calcLocalTerm(int &row, const vector<matchWithDst_t> &matches,
                               const vector<ImageFeatures> &features, int idx)
{
    float f = focal_length;
    float a = sqrt(ALPHAS[0]);
    for (int m = 0; m < matches.size(); ++m) {
        const matchWithDst_t matchWDst = matches.at(m);
        DMatch match = matches.at(m).match;

        int queryIdx = match.queryIdx;
        int trainIdx = match.trainIdx;
        int src = idx;
        int dst = matchWDst.dst;

        float h1 = features[src].img_size.height;
        float w1 = features[src].img_size.width;
        float h2 = features[dst].img_size.height;
        float w2 = features[dst].img_size.width;

        // Distance between dst and src in radians
        float theta = dst - src;
        if (src == 0 && dst == NUM_IMAGES - 1 && wrapAround) {
            theta = -1;
        }
        // Hardcoded values and camera sources. Opencv splits the third video in the middle
        if (src == 3) {
            theta = 4.25f;
        }
        if (src == 4) {
            theta = -0.25f;
        }

        theta *= 2 * PI / 6;
        Point2f p1 = features[src].keypoints[queryIdx].pt;
        Point2f p2 = features[dst].keypoints[trainIdx].pt;

        float scale = compose_scale / work_scale;
        float x1_ = p1.x;
        float y1_ = p1.y;
        float x2_ = p2.x;
        float y2_ = p2.y;

        // Ignore features which have been warped outside of either image
        if (x1_ < 0 || x2_ < 0 || y1_ < 0 || y2_ < 0 || x1_ >= w1 || x2_ >= w2
                || y1_ >= h1 || y2_ >= h2 ) {
            continue;
        }

        // Calculate in which rectangle the features are
        int t1 = floor(y1_ * (N-1) / h1);
        int t2 = floor(y2_ * (N-1) / h2);
        int l1 = floor(x1_ * (M-1) / w1);
        int l2 = floor(x2_ * (M-1) / w2);

        // Calculate coordinates for the corners of the recatngles the features are in
        float top1 = t1 * h1 / (N-1);
        float bot1 = top1 + h1 / (N-1); // (t1+1) * h1 / N
        float left1 = l1 * w1 / (M-1);
        float right1 = left1 + w1 / (M-1); // (l1+1) * w1 / M
        float top2 = t2 * h2 / (N-1);
        float bot2 = top2 + h2 / (N-1); // (t2+1) * h2 / N
        float left2 = l2 * w2 / (M-1);
        float right2 = left2 + w2 / (M-1); // (l2+1) * w2 / M

        // Calculate local coordinates for the features within the rectangles
        float u1 = (x1_ - left1) / (right1 - left1);
        float u2 = (x2_ - left2) / (right2 - left2);
        float v1 = (y1_ - top1) / (bot1 - top1);
        float v2 = (y2_ - top2) / (bot2 - top2);


        // _x_ - _x2_ = theta * f * scale
        // Differs from the way of warping in the paper.
        // This constraint tries keep x-distance between featurepoint fp1 and featurepoint fp2
        // constant across the whole image. The desired x-distance constant between featurepoints is
        // theta * f * scale * a. 
        // For example if "theta * f * scale * a" is replaced with "fp1 - fp2", the warping will do nothing

        // fp1 bilinear mapping
        // from: https://www2.eecs.berkeley.edu/Pubs/TechRpts/1989/CSD-89-516.pdf
        A.insert(row,  2*(l1   + M * (t1)   + M*N*src)) = (1-u1)*(1-v1) * a;
        A.insert(row,  2*(l1+1 + M * (t1)   + M*N*src)) = u1*(1-v1) * a;
        A.insert(row,  2*(l1 +   M * (t1+1) + M*N*src)) = v1*(1-u1) * a;
        A.insert(row,  2*(l1+1 + M * (t1+1) + M*N*src)) = u1*v1 * a;
        // fp2 bilinear mapping
        A.insert(row,  2*(l2   + M * (t2)   + M*N*dst)) = -(1-u2)*(1-v2) * a;
        A.insert(row,  2*(l2+1 + M * (t2)   + M*N*dst)) = -u2*(1-v2) * a;
        A.insert(row,  2*(l2 +   M * (t2+1) + M*N*dst)) = -v2*(1-u2) * a;
        A.insert(row,  2*(l2+1 + M * (t2+1) + M*N*dst)) = -u2*v2 * a;
        // distance to warp to the feature points
        b(row) = theta * f * scale * a;


        // _y_ - _y2_ = 0
        // Differs from the way of warping in the paper.
        // This constraint tries keep y-distance between featurepoint fp1 and featurepoint fp2
        // constant across the whole image. The desired y-distance constant between featurepoints is 0.

        // fp1 bilinear mapping
        A.insert(row+1, 2*(l1   + M * (t1)   + M*N*src)+1) = (1-u1)*(1-v1) * a;
        A.insert(row+1, 2*(l1+1 + M * (t1)   + M*N*src)+1) = u1*(1-v1) * a;
        A.insert(row+1, 2*(l1 +   M * (t1+1) + M*N*src)+1) = v1*(1-u1) * a;
        A.insert(row+1, 2*(l1+1 + M * (t1+1) + M*N*src)+1) = u1*v1 * a;
        // fp2 bilinear mapping
        A.insert(row+1, 2*(l2   + M * (t2)   + M*N*dst)+1) = -(1-u2)*(1-v2) * a;
        A.insert(row+1, 2*(l2+1 + M * (t2)   + M*N*dst)+1) = -u2*(1-v2) * a;
        A.insert(row+1, 2*(l2 +   M * (t2+1) + M*N*dst)+1) = -v2*(1-u2) * a;
        A.insert(row+1, 2*(l2+1 + M * (t2+1) + M*N*dst)+1) = -u2*v2 * a;
        // b should be zero anyway, but for posterity's sake set it to zero
        b(row + 1) = 0;

        row+=2;
    }
}

Mat MeshWarper::drawMesh(const Mat &mesh_cpu_x, const Mat &mesh_cpu_y, Size mesh_size)
{
    Mat mat(mesh_size.height, mesh_size.width, CV_16UC3);
    mat.setTo(Scalar::all(255 * 255));
    for (int i = 0; i < mesh_cpu_x.rows - 1; ++i) {
        for (int j = 0; j < mesh_cpu_x.cols; ++j) {
            Point start = Point(mesh_cpu_x.at<float>(i, j), mesh_cpu_y.at<float>(i, j));
            Point end = Point(mesh_cpu_x.at<float>(i + 1, j), mesh_cpu_y.at<float>(i + 1, j));
            line(mat, start, end, Scalar(1, 0, 0), 3);
        }
    }
    for (int i = 0; i < mesh_cpu_x.rows; ++i) {
        for (int j = 0; j < mesh_cpu_x.cols - 1; ++j) {
            Point start = Point(mesh_cpu_x.at<float>(i, j), mesh_cpu_y.at<float>(i, j));
            Point end = Point(mesh_cpu_x.at<float>(i, j + 1), mesh_cpu_y.at<float>(i, j + 1));
            line(mat, start, end, Scalar(1, 0, 0), 3);
        }
    }
    return mat;
}


void MeshWarper::convertVectorToMesh(const Eigen::VectorXd &x, Mat &out_mesh_x, Mat &out_mesh_y, int idx)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            out_mesh_x.at<float>(i, j) = x(2 * (j + i * M + idx * M * N));
            out_mesh_y.at<float>(i, j) = x(2 * (j + i * M + idx * M * N) + 1);
        }
    }
}

// Convert the mesh into a backward map used by opencv remap function
// @TODO implement this in CUDA so it can be run entirely on GPU
void MeshWarper::convertMeshToMap(cv::Mat &mesh_cpu_x, cv::Mat &mesh_cpu_y, cuda::GpuMat &map_x, cuda::GpuMat &map_y, cv::Size mesh_size)
{
    cuda::GpuMat gpu_small_mesh_x;
    cuda::GpuMat gpu_small_mesh_y;
    cuda::GpuMat big_x;
    cuda::GpuMat big_y;
    Mat big_mesh_x;
    Mat big_mesh_y;
    // Interpolate pixel positions between the mesh vertices with a custom resize function
    gpu_small_mesh_x.upload(mesh_cpu_x);
    gpu_small_mesh_y.upload(mesh_cpu_y);
    custom_resize(gpu_small_mesh_x, big_x, mesh_size);
    custom_resize(gpu_small_mesh_y, big_y, mesh_size);
    big_x.download(big_mesh_x);
    big_y.download(big_mesh_y);

    // Calculate pixel values for a map half the width and height and then resize the map back to
    // full size
    int scale = 2;
    Mat warp_x(mesh_size.height / scale, mesh_size.width / scale, mesh_cpu_x.type());
    Mat warp_y(mesh_size.height / scale, mesh_size.width / scale, mesh_cpu_y.type());
    Mat set_values(mesh_size.height / scale, mesh_size.width / scale, CV_32F);
    set_values.setTo(Scalar::all(0));
    warp_x.setTo(Scalar::all(0));
    warp_y.setTo(Scalar::all(0));

    Mat sum_x(mesh_size.height / scale, mesh_size.width / scale, mesh_cpu_x.type());
    Mat sum_y(mesh_size.height / scale, mesh_size.width / scale, mesh_cpu_y.type());
    sum_x.setTo(Scalar::all(0));
    sum_y.setTo(Scalar::all(0));


    for (int y = 0; y < mesh_size.height; y++) {
        for (int x = 0; x < mesh_size.width; x++) {
            int x_ = (int)big_mesh_x.at<float>(y, x) / scale;
            int y_ = (int)big_mesh_y.at<float>(y, x) / scale;
            if (x_ >= 0 && y_ >= 0 && x_ < warp_x.cols && y_ < warp_x.rows) {
                    sum_x.at<float>(y_, x_) += (float)x;
                    sum_y.at<float>(y_, x_) += (float)y;
                    set_values.at<float>(y_, x_)++;
            }
        }
    }
    for (int y = 0; y < sum_x.rows; ++y) {
        for (int x = 0; x < sum_x.cols; ++x) {
                warp_x.at<float>(y, x) = (float)sum_x.at<float>(y, x) / (float)set_values.at<float>(y, x);
                warp_y.at<float>(y, x) = (float)sum_y.at<float>(y, x) / (float)set_values.at<float>(y, x);
        }
    }

    gpu_small_mesh_x.upload(warp_x);
    gpu_small_mesh_y.upload(warp_y);
    custom_resize(gpu_small_mesh_x, map_x, mesh_size);
    custom_resize(gpu_small_mesh_y, map_y, mesh_size);
}
