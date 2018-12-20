#include "meshwarp.h"
#include "featurefinder.h"
#include "debug.h"

#include "opencv2/imgproc.hpp"




using namespace cv;
using namespace cv::detail;
using std::vector;


void meshwarp::calibrateMeshWarp(vector<Mat> &full_imgs, vector<ImageFeatures> &features,
                       vector<MatchesInfo> &pairwise_matches, vector<cuda::GpuMat> &x_mesh,
                       vector<cuda::GpuMat> &y_mesh, vector<cuda::GpuMat> &x_maps,
                       vector<cuda::GpuMat> &y_maps, float focal_length, double compose_scale,
                       double work_scale) {
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
                mesh_cpu_x[idx].at<float>(i, j) = static_cast<float>(j) * mesh_size[idx].width / (M-1);
                mesh_cpu_y[idx].at<float>(i, j) = static_cast<float>(i) * mesh_size[idx].height / (N-1);
            }
        }
    }

    featurefinder::findFeatures(images, features, -1);
    featurefinder::matchFeatures(features, pairwise_matches);

    int features_per_image = MAX_FEATURES_PER_IMAGE;
    // 2 * images.size()*N*M for global alignment, 2*images.size()*(N-1)*(M-1) for smoothness term and
    // 2 * (NUM_IMAGES +(int)wrapAround + 1) * features_per_image for local alignment
    int num_rows = 2 * static_cast<int>(images.size())*N*M + 2* static_cast<int>(images.size())*(N-1)*(M-1)*8 +
                   2 * (NUM_IMAGES + (int)wrapAround + 1) * features_per_image;
    Eigen::SparseMatrix<double> A(num_rows, 2*N*M*images.size());
    Eigen::VectorXd b(num_rows);
    Eigen::VectorXd x;
    b.fill(0);

    int global_start = 0;
    int smooth_start = 2 * static_cast<int>(images.size())*N*M;
    int local_start = 2 * static_cast<int>(images.size())*N*M + 2* static_cast<int>(images.size())*(N-1)*(M-1)*8;


    vector<int> valid_indexes_orig_all[NUM_IMAGES];
    vector<DMatch> all_matches[NUM_IMAGES];

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
            Mat out;
            drawMatches(images[src], features[src].keypoints, images[dst], features[dst].keypoints, pw_matches.matches, out);
            showMat("out", out);
        }

        //Find all indexes of the inliers_mask that contain the value 1
        for (int i = 0; i < pw_matches.inliers_mask.size(); ++i) {
            if (pw_matches.inliers_mask[i]) {
                valid_indexes_orig_all[src].push_back(i);
                DMatch match = pw_matches.matches[i];
                all_matches[src].push_back(match);
            }
        }
    }


    vector<int> valid_indexes_selected[NUM_IMAGES];
    vector<DMatch> selected_matches[NUM_IMAGES];

    // Select features_per_image amount of random features points from valid_indexes_orig_all
    for (int img = 0; img < NUM_IMAGES; ++img) {
        vector<int> valid_indexes;
        valid_indexes = valid_indexes_orig_all[img];
        //Shuffle the index vector on each loop to get random results each time
        std::random_shuffle(valid_indexes.begin(), valid_indexes.end());

        for (int i = 0; i < min(features_per_image, (int)(valid_indexes.size() * 0.8f)); ++i) {
            int idx = valid_indexes.at(i);
            valid_indexes_selected[img].push_back(idx);

            DMatch match = all_matches[img].at(i);
            selected_matches[img].push_back(match);
        }
    }


    // Global alignment term from http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
    // Square root ALPHAS, because equation is in format |Ax + b|^2 instead of Ax + b
    // |sqrt(ALPHA) * (Ax + b)|^2 == ALPHA * |Ax + b|^2
    float a = sqrt(ALPHAS[1]);
    int row = 0;
    for (int idx = 0; idx < images.size(); ++idx) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                float x1 = static_cast<float>(j * mesh_size[idx].width / (M-1));
                float y1 = static_cast<float>(i * mesh_size[idx].height / (N-1));
                float scale = compose_scale / work_scale;
                float tau = 1;

                for (int ft = 0; ft < selected_matches[idx].size(); ++ft) {
                    int ft_id = selected_matches[idx][ft].queryIdx;
                    Point ft_point = features[idx].keypoints[ft_id].pt;
                    if (sqrt(pow(ft_point.x - x1, 2) + pow(ft_point.y - y1, 2)) < GLOBAL_DIST) {
                        tau = 0;
                        if (VISUALIZE_MATCHES)
                            circle(images[idx], Point(ft_point.x, ft_point.y), 4, Scalar(0, 255, 255), 3);
                        else
                            break;
                    }
                }

                A.insert(row, row) = a * tau;
                A.insert(row + 1, row + 1) = a * tau;
                b(row) = a * tau * x1;
                b(row + 1) = a * tau * y1;
                row += 2;
            }
        }
    }

    row = 0;
    // Smoothness term from http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
    a = sqrt(ALPHAS[2]);
    for (int idx = 0; idx < images.size(); ++idx) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                // Loop through all triangles surrounding vertex (j, i), when the surrounding quads are split in the middle
                // Left to right, top to bottom order
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

                    float V1x = mesh_cpu_x[idx].at<float>(Vi_total[0].y, Vi_total[0].x);
                    float V2x = mesh_cpu_x[idx].at<float>(Vi_total[1].y, Vi_total[1].x);
                    float V3x = mesh_cpu_x[idx].at<float>(Vi_total[2].y, Vi_total[2].x);

                    float V1y = mesh_cpu_y[idx].at<float>(Vi_total[0].y, Vi_total[0].x);
                    float V2y = mesh_cpu_y[idx].at<float>(Vi_total[1].y, Vi_total[1].x);
                    float V3y = mesh_cpu_y[idx].at<float>(Vi_total[2].y, Vi_total[2].x);

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
                    Mat mask(images[idx].rows, images[idx].cols, CV_8UC1);
                    mask.setTo(Scalar::all(0));
                    Point pts[3] = { Point(V1x, V1y), Point(V2x, V2y), Point(V3x, V3y) };
                    fillConvexPoly(mask, pts, 3, Scalar(255));

                    Mat mean;
                    Mat deviation;

                    // meanstdDev is a very slow operation. Make it faster
                    meanStdDev(images[idx], mean, deviation, mask);
                    Mat variance;
                    pow(deviation, 2, variance);
                    sal = sqrt(norm(variance, NORM_L2) + 0.5f);



                    A.insert(smooth_start + row,   2*((j + Vi[0].x) + M * (i + Vi[0].y) + M*N*idx)) = a * sal; // x1
                    A.insert(smooth_start + row,   2*((j + Vi[0].x) + M * (i + Vi[0].y) + M*N*idx) + 1) = a * sal; // y1
                    A.insert(smooth_start + row,   2*((j + Vi[1].x) + M * (i + Vi[1].y) + M*N*idx)) = a*(u - v - 1) * sal; // x2
                    A.insert(smooth_start + row,   2*((j + Vi[1].x) + M * (i + Vi[1].y) + M*N*idx) + 1) = a*(u + v - 1) * sal; // y2
                    A.insert(smooth_start + row,   2*((j + Vi[2].x) + M * (i + Vi[2].y) + M*N*idx)) = a*(-u + v) * sal; // x3
                    A.insert(smooth_start + row,   2*((j + Vi[2].x) + M * (i + Vi[2].y) + M*N*idx) + 1) = a*(-u - v) * sal; // y3

                    // b should be zero anyway, but for posterity's sake set it to zero
                    b(smooth_start + row) = 0;


                    A.insert(smooth_start + row + 1,   2*((j + Vi[0].x) + M * (i + Vi[0].y) + M*N*idx)) = a * sal; // x1
                    A.insert(smooth_start + row + 1,   2*((j + Vi[0].x) + M * (i + Vi[0].y) + M*N*idx) + 1) = a * sal; // y1
                    A.insert(smooth_start + row + 1,   2*((j + Vi[1].x) + M * (i + Vi[1].y) + M*N*idx)) = a*(u - v - 1) * sal; // x2
                    A.insert(smooth_start + row + 1,   2*((j + Vi[1].x) + M * (i + Vi[1].y) + M*N*idx) + 1) = a*(u + v - 1) * sal; // y2
                    A.insert(smooth_start + row + 1,   2*((j + Vi[2].x) + M * (i + Vi[2].y) + M*N*idx)) = a*(-u + v) * sal; // x3
                    A.insert(smooth_start + row + 1,   2*((j + Vi[2].x) + M * (i + Vi[2].y) + M*N*idx) + 1) = a*(-u - v) * sal; // y3

                    // b should be zero anyway, but for posterity's sake set it to zero
                    b(smooth_start + row + 1) = 0;
                    row += 2;
                }
            }
        }
    }

    if (VISUALIZE_MATCHES) { // Draw the meshes for visualisation
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

    // Local alignment term from http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
    row = 0;
    float f = focal_length;
    a = sqrt(ALPHAS[0]);
    vector<int> valid_indexes_orig;
    vector<int> valid_indexes;
    for (int idx = 0; idx < pairwise_matches.size(); ++idx) {
        MatchesInfo &pw_matches = pairwise_matches[idx];
        if (!pw_matches.matches.size() || !pw_matches.num_inliers) continue;
        int src = pw_matches.src_img_idx;
        int dst = pw_matches.dst_img_idx;

        valid_indexes_orig = valid_indexes_selected[src];
        if (dst != NUM_IMAGES - 1 || src != 0) {
            // Only calculate loss from pairs of src and dst where src = dst - 1
            // to avoid using same pairs multiple times
            if (abs(src - dst - 1) > 0.1) {
                continue;
            }
        }

        valid_indexes = valid_indexes_orig;
        for(int i = 0; i < valid_indexes.size(); ++i) {
            int idx = valid_indexes.at(i);

            int idx1 = pw_matches.matches[idx].queryIdx;
            int idx2 = pw_matches.matches[idx].trainIdx;
            KeyPoint k1 = features[src].keypoints[idx1];
            KeyPoint k2 = features[dst].keypoints[idx2];

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
            Point2f p1 = features[src].keypoints[idx1].pt;
            Point2f p2 = features[dst].keypoints[idx2].pt;

            float scale = compose_scale / work_scale;

            float x1_ = p1.x;
            float y1_ = p1.y;
            float x2_ = p2.x;
            float y2_ = p2.y;

            // change the image sizes to compose scale as well
            h1 = images[src].rows;
            w1 = images[src].cols;
            h2 = images[dst].rows;
            w2 = images[dst].cols;

            // Ignore features which have been warped outside of either image
            if (x1_ < 0 || x2_ < 0 || y1_ < 0 || y2_ < 0 || x1_ >= w1 || x2_ >= w2
                    || y1_ >= h1 || y2_ >= h2 ) {
                continue;
            }

            if(VISUALIZE_MATCHES) {
                circle(images[src], Point(x1_, y1_), 3, Scalar(0, 255, 0), 2);
                circle(images[dst], Point(x2_, y2_), 3, Scalar(0, 0, 255), 2);
                imshow(std::to_string(src), images[src]);
                imshow(std::to_string(dst), images[dst]);
                waitKey(0);
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
            A.insert(local_start + row,  2*(l1   + M * (t1)   + M*N*src)) = (1-u1)*(1-v1) * a;
            A.insert(local_start + row,  2*(l1+1 + M * (t1)   + M*N*src)) = u1*(1-v1) * a;
            A.insert(local_start + row,  2*(l1 +   M * (t1+1) + M*N*src)) = v1*(1-u1) * a;
            A.insert(local_start + row,  2*(l1+1 + M * (t1+1) + M*N*src)) = u1*v1 * a;
            // fp2 bilinear mapping
            A.insert(local_start + row,  2*(l2   + M * (t2)   + M*N*dst)) = -(1-u2)*(1-v2) * a;
            A.insert(local_start + row,  2*(l2+1 + M * (t2)   + M*N*dst)) = -u2*(1-v2) * a;
            A.insert(local_start + row,  2*(l2 +   M * (t2+1) + M*N*dst)) = -v2*(1-u2) * a;
            A.insert(local_start + row,  2*(l2+1 + M * (t2+1) + M*N*dst)) = -u2*v2 * a;
            // distance to warp to the feature points
            b(local_start + row) = theta * f * scale * a;


            // _y_ - _y2_ = 0
            // Differs from the way of warping in the paper.
            // This constraint tries keep y-distance between featurepoint fp1 and featurepoint fp2
            // constant across the whole image. The desired y-distance constant between featurepoints is 0.

            // fp1 bilinear mapping
            A.insert(local_start + row+1, 2*(l1   + M * (t1)   + M*N*src)+1) = (1-u1)*(1-v1) * a;
            A.insert(local_start + row+1, 2*(l1+1 + M * (t1)   + M*N*src)+1) = u1*(1-v1) * a;
            A.insert(local_start + row+1, 2*(l1 +   M * (t1+1) + M*N*src)+1) = v1*(1-u1) * a;
            A.insert(local_start + row+1, 2*(l1+1 + M * (t1+1) + M*N*src)+1) = u1*v1 * a;
            // fp2 bilinear mapping
            A.insert(local_start + row+1, 2*(l2   + M * (t2)   + M*N*dst)+1) = -(1-u2)*(1-v2) * a;
            A.insert(local_start + row+1, 2*(l2+1 + M * (t2)   + M*N*dst)+1) = -u2*(1-v2) * a;
            A.insert(local_start + row+1, 2*(l2 +   M * (t2+1) + M*N*dst)+1) = -v2*(1-u2) * a;
            A.insert(local_start + row+1, 2*(l2+1 + M * (t2+1) + M*N*dst)+1) = -u2*v2 * a;
            // b should be zero anyway, but for posterity's sake set it to zero
            b(local_start + row + 1) = 0;

            row+=2;
        }
    }

    // LeastSquaresConjudateGradientSolver solves equations that are in format |Ax + b|^2
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    x = solver.solve(b);


    for (int idx = 0; idx < full_imgs.size(); ++idx) {
        convertVectorToMesh(x, mesh_cpu_x[idx], mesh_cpu_y[idx], idx);
        convertMeshToMap(mesh_cpu_x[idx], mesh_cpu_y[idx], x_mesh[idx], y_mesh[idx], mesh_size[idx]);

        if (VISUALIZE_WARPED) {
            Mat visualized_mesh = meshwarp::drawMesh(mesh_cpu_x[idx], mesh_cpu_y[idx], mesh_size[idx]);
            imshow(std::to_string(idx), visualized_mesh);
            waitKey();
        }
    }
}

void meshwarp::recalibrateMesh(std::vector<cv::Mat> &full_img, std::vector<cv::cuda::GpuMat> &x_maps,
                     std::vector<cv::cuda::GpuMat> &y_maps, std::vector<cv::cuda::GpuMat> &x_mesh,
                     std::vector<cv::cuda::GpuMat> &y_mesh, float focal_length, double compose_scale,
                     const double &work_scale)
{
    vector<ImageFeatures> features(NUM_IMAGES);
    vector<MatchesInfo> pairwise_matches;

    meshwarp::calibrateMeshWarp(full_img, features, pairwise_matches, x_mesh, y_mesh, x_maps, y_maps,
                      focal_length, compose_scale, work_scale);
}

Mat meshwarp::drawMesh(const Mat &mesh_cpu_x, const Mat &mesh_cpu_y, Size mesh_size)
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


//TODO: Don't use global N and M
void meshwarp::convertVectorToMesh(const Eigen::VectorXd &x, Mat &out_mesh_x, Mat &out_mesh_y, int idx)
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
void meshwarp::convertMeshToMap(cv::Mat &mesh_cpu_x, cv::Mat &mesh_cpu_y, cuda::GpuMat &map_x, cuda::GpuMat &map_y, cv::Size mesh_size)
{
    cuda::GpuMat gpu_small_mesh_x;
    cuda::GpuMat gpu_small_mesh_y;
    cuda::GpuMat big_x;
    cuda::GpuMat big_y;
    Mat big_mesh_x;
    Mat big_mesh_y;
    // Interpolate pixel positions between the mesh vertices by using a custom resize function
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
