#include "meshwarper.h"
#include "featurefinder.h"
#include "debug.h"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/cudawarping.hpp"

#include "opencv2/imgproc.hpp"




using namespace cv;
using namespace cv::detail;
using std::vector;
extern void custom_resize(cv::cuda::GpuMat &in, cv::cuda::GpuMat &out, cv::Size t_size);


MeshWarper::MeshWarper(int num_images, int mesh_width, int mesh_height, 
                       float focal_length, double compose_scale, double work_scale):
    M(mesh_width), N(mesh_height), focal_length(focal_length),
    compose_scale(compose_scale), work_scale(work_scale),  old_features(NUM_IMAGES)
{
    int features_per_image = MAX_FEATURES_PER_IMAGE;
    // 2 dimensions, x and y
    int dims = 2;

    int global_size = dims * num_images * N * M;
    // N - 1 and M - 1, since edges will only require half compared to the middle
    int smoothness_size = dims * num_images * (N - 1) * (M - 1) * 8;
    int local_size = dims * (num_images + (int)wrapAround + 1) * features_per_image;

    int local_temporal_size = 0;
    if (USE_TEMPORAL)
        local_temporal_size = dims * (num_images + (int)wrapAround + 1) * MAX_TEMPORAL_FEATURES_PER_IMAGE;

    // Number of rows/equations needed for calculating the mesh
    int num_rows = global_size + smoothness_size + local_size + local_temporal_size;
    A.resize(num_rows, dims * N * M * num_images);
    b.resize(num_rows);
    b.fill(0);
}

void MeshWarper::createMesh(LockableVector<cv::Mat> &full_imgs, std::vector<cv::detail::ImageFeatures> &features,
                           std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                           std::vector<cv::Mat> &mesh_cpu_x, std::vector<cv::Mat> &mesh_cpu_y,
                           std::vector<cv::cuda::GpuMat> &x_maps, std::vector<cv::cuda::GpuMat> &y_maps,
                           std::vector<cv::Size> &mesh_size) {
    A.setZero();
    b.fill(0);
    full_imgs.lock();
    int full_imgs_count = full_imgs.size();
    full_imgs.unlock();
    if (full_imgs_count != NUM_IMAGES)
        return;
    vector<Mat> images(full_imgs_count);
    vector<Mat> x_map(full_imgs_count);
    vector<Mat> y_map(full_imgs_count);

    for (int idx = 0; idx < full_imgs_count; ++idx) {
        mesh_cpu_x[idx] = Mat(N, M, CV_32FC1);
        mesh_cpu_y[idx] = Mat(N, M, CV_32FC1);
        full_imgs.lock();
        resize(full_imgs.at(idx), images.at(idx), Size(), compose_scale, compose_scale);
        full_imgs.unlock();
        x_maps[idx].download(x_map[idx]);
        y_maps[idx].download(y_map[idx]);
        remap(images[idx], images[idx], x_map[idx], y_map[idx], INTER_LINEAR);
        mesh_size[idx] = images[idx].size();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                mesh_cpu_x[idx].at<float>(i, j) = static_cast<float>(j) * mesh_size[idx].width / (M - 1);
                mesh_cpu_y[idx].at<float>(i, j) = static_cast<float>(i) * mesh_size[idx].height / (N - 1);
            }
        }
    }

    vector<Mat> feature_mask(full_imgs_count);
    for (int idx = 0; idx < NUM_IMAGES; ++idx) {
        if (use_surf) { //TODO: masks don't seem to work on surf
            continue;
        }
        feature_mask.at(idx) = Mat(images.at(idx).rows, images.at(idx).cols, CV_8U);
        feature_mask.at(idx).setTo(Scalar::all(0));
        //TODO: calculate overlap
        //Overlap will change based on distance distance of the subject from the cameras
        //closer = less overlap, further = more overlap
        int overlap = 400;

        int img_cols = images.at(idx).cols;
        int img_rows = images.at(idx).rows;
        // Hardcoded values and camera sources. Opencv splits the third video(idx==3) in the middle
        if (idx == 3) {
            float split_offset_l = images.at(0).cols * 0.25f * 2.0f;
            float split_offset_r = -images.at(0).cols * 0.25f * 2.0f;

            Rect r = {(int)split_offset_l - overlap, 0, overlap, img_rows};
            rectangle(feature_mask.at(idx), r, Scalar::all(255), CV_FILLED);

            r = {img_cols + (int)split_offset_r, 0, overlap, img_rows};
            rectangle(feature_mask.at(idx), r, Scalar::all(255), CV_FILLED);
        } else {
            Rect r = {0, 0, overlap, img_rows};
            rectangle(feature_mask.at(idx), r, Scalar::all(255), CV_FILLED);
            r = {img_cols - overlap, 0, overlap, img_rows};
            rectangle(feature_mask.at(idx), r, Scalar::all(255), CV_FILLED);
        }
        // add image mask to prevent finding features of the image's borders
        Mat image_mask = Mat(images.at(idx).rows, images.at(idx).cols, CV_8U);
        cvtColor(images[idx], image_mask, CV_8U);
        inRange(images[idx], Scalar::all(0), Scalar::all(0), image_mask);
        bitwise_not(image_mask, image_mask);
        bitwise_and(image_mask, feature_mask.at(idx), feature_mask.at(idx));
    }


    featurefinder::findFeatures(images, feature_mask, features, -1);
    featurefinder::matchFeatures(features, pairwise_matches);

    vector<MatchesInfo> temporal_matches(NUM_IMAGES);
    vector<matchWithDst_t> selected_temp_matches[NUM_IMAGES];
    if (USE_TEMPORAL) {
        if (!prev_features.empty()) {
            featurefinder::matchFeaturesTemporal(features, prev_features, temporal_matches);
            for (int idx = 0; idx < temporal_matches.size(); ++idx) {
                MatchesInfo &pw_matches = temporal_matches[idx];
                if (VISUALIZE_TEMPORAL_MATCHES) {
                    Mat visual_matches;
                    drawMatches(images[idx], features[idx].keypoints, images[idx], prev_features[idx].keypoints, pw_matches.matches, visual_matches);
                    //lock full_imgs to stop drawing frames using the old calibration
                    full_imgs.lock();
                    showMat("visualized matches", visual_matches);
                    full_imgs.unlock();
                }
            }
        }

        vector<matchWithDst_t> all_temp_matches[NUM_IMAGES];
        filterTemporalMatches(temporal_matches, features, all_temp_matches);


        int temporal_feat_per_image = MAX_TEMPORAL_FEATURES_PER_IMAGE;
        // Select features_per_image amount of random features points from all_matches
        for (int img = 0; img < NUM_IMAGES; ++img) {
            for (int i = 0; i < min(temporal_feat_per_image, (int)(all_temp_matches[img].size())); ++i) {
                matchWithDst_t match = all_temp_matches[img].at(i);
                selected_temp_matches[img].push_back(match);
            }
        }
    }


    int features_per_image = MAX_FEATURES_PER_IMAGE;

    vector<matchWithDst_t> all_matches[NUM_IMAGES];
    filterMatches(pairwise_matches, features, all_matches);

    if (VISUALIZE_MATCHES) {
        for (int idx = 0; idx < pairwise_matches.size(); ++idx) {
            MatchesInfo &pw_matches = pairwise_matches[idx];
            int src = pw_matches.src_img_idx;
            int dst = pw_matches.dst_img_idx;
            Mat visual_matches;
            drawMatches(images[src], features[src].keypoints, images[dst], features[dst].keypoints, pw_matches.matches, visual_matches);
            //lock full_imgs to stop drawing frames using the old calibration
            full_imgs.lock();
            showMat("visualized matches", visual_matches);
            full_imgs.unlock();
        }
    }



    vector<matchWithDst_t> selected_matches[NUM_IMAGES];

    // Select features_per_image amount of random features points from all_matches
    for (int img = 0; img < NUM_IMAGES; ++img) {
        /* std::sort(all_matches[img].begin(), all_matches[img].end(), */ 
        /* [](matchWithDst_t a, matchWithDst_t b){ */
        /*     return a.match.distance < b.match.distance; */
        /* }); */
        for (int i = 0; i < min(features_per_image, (int)(all_matches[img].size())); ++i) {
            matchWithDst_t match = all_matches[img].at(i);
            selected_matches[img].push_back(match);
        }
    }

    vector<Point> selected_points[NUM_IMAGES];
    // Convert matchWithDst_t to points, which is needed for the global term
    for (int idx = 0; idx < NUM_IMAGES; ++idx) {
        for (int i = 0; i < selected_matches[idx].size(); ++i) {
            int ft_id = selected_matches[idx].at(i).match.queryIdx;
            Point p = features[idx].keypoints.at(ft_id).pt;
            selected_points[idx].push_back(p);
        }
    }

    vector<Point> old_selected_points[NUM_IMAGES];
    // Convert matchWithDst_t to points, which is needed for the global term,
    // use old points if featurepoints haven't moved significantly
    for (int idx = 0; idx < NUM_IMAGES; ++idx) {
        for (int i = 0; i < old_matches[idx].size(); ++i) {
            int ft_id = old_matches[idx].at(i).match.queryIdx;
            Point p = old_features.at(idx)[idx].keypoints.at(ft_id).pt;
            old_selected_points[idx].push_back(p);
        }
    }



    // fp_{sum,count,avg} are saved in format:
    // [idx * 2] means feature points in left side of the image with id=idx
    // [idx * 2 + 1] means feature points in right side of the image with id=idx
    float fp_sum[NUM_IMAGES * 2] = {0};
    float fp_count[NUM_IMAGES * 2] = {0};
    float fp_avg[NUM_IMAGES * 2] = {0};


    // Calculate average x-coordinate of the feature points for each image half
    for (int idx = 0; idx < NUM_IMAGES; ++idx) {
        for (int i = 0; i < selected_matches[idx].size(); ++i) {
            DMatch m = selected_matches[idx].at(i).match;
            int dst = selected_matches[idx].at(i).dst;
            int src = idx;
            int queryIdx = m.queryIdx;
            int trainIdx = m.trainIdx;
            Point2f p1 = features[src].keypoints[queryIdx].pt;
            Point2f p2 = features[dst].keypoints[trainIdx].pt;

            fp_sum[src * 2] += p1.x;
            fp_sum[dst * 2 + 1] += p2.x;
            fp_count[src * 2]++;
            fp_count[dst * 2 + 1]++;
        }
    }

    for (int idx = 0; idx < NUM_IMAGES; ++idx) {
        fp_avg[idx * 2] = 0;
        fp_avg[idx * 2 + 1] = 0;
        if (fp_count[idx * 2] != 0)
            fp_avg[idx * 2] = fp_sum[idx * 2] / fp_count[idx * 2];

        if (fp_count[idx * 2 + 1] != 0)
            fp_avg[idx * 2 + 1] = fp_sum[idx * 2 + 1] / fp_count[idx * 2 + 1];
    }


    // Decide whether features have moved significantly and need to be updated
    bool use_old_features[6];
    for (int idx = 0; idx < NUM_IMAGES; ++idx) {
        use_old_features[idx] = false;
        if (old_features.empty())
            break;

        // Image left of idx
        int idx2 = (idx - 1 == -1) ? NUM_IMAGES - 1 : idx-1;

        // Calculate average distance of matched feature points
        float avg_diff = abs(fp_avg[idx * 2] - fp_avg[idx2 * 2 + 1]);
        float avg_diff_prev = abs(prev_avg[idx * 2] - prev_avg[idx2 * 2 + 1]);

        bool found_matches = fp_avg[idx * 2] != 0 && fp_avg[idx2 * 2 + 1] != 0;
        bool found_prev_matches = prev_avg[idx * 2] != 0 && prev_avg[idx2 * 2 + 1] != 0;

        // if average distance of matched features have moved more than RECALIB_THRESH
        if ((abs(avg_diff - avg_diff_prev) < RECALIB_THRESH) || (!found_matches && found_prev_matches)) {
            use_old_features[idx] = true;
        } else {
            /*
            LOGLN("idx " << idx << " DIST: " << abs(avg_diff_prev - avg_diff));
            LOGLN("found: " << found_matches << " prev_found: " << found_prev_matches);
            LOGLN("fp count" << fp_count[idx * 2] << " " << fp_count[idx2 * 2 + 1]);
            LOGLN("avg_diff:" << avg_diff << " diff_prev:" << avg_diff_prev);
            LOGLN("fp_avg:" << fp_avg[idx * 2] << " " << fp_avg[idx2 * 2 + 1] <<
            " prev_avg:" << prev_avg[idx * 2] << " " << prev_avg[idx2 * 2 + 1]);
            */
        }

    }


    int rows_used = 0;
    for (int idx = 0; idx < NUM_IMAGES; ++idx) {
        if (use_old_features[idx]) {
            calcLocalTerm(rows_used, old_matches[idx], old_features.at(idx), idx);
            calcGlobalTerm(rows_used, old_selected_points[idx], mesh_size.at(idx), idx);
        } else  {
            calcLocalTerm(rows_used, selected_matches[idx], features, idx);
            calcGlobalTerm(rows_used, selected_points[idx], mesh_size.at(idx), idx);
        }
        calcSmoothnessTerm(rows_used, images.at(idx), mesh_size.at(idx), idx);
        if (USE_TEMPORAL) {
            calcTemporalLocalTerm(rows_used, selected_temp_matches[idx], features, idx);
        }
    }

    // LeastSquaresConjudateGradientSolver solves equations that are in format |Ax + b|^2
    Eigen::VectorXd x;
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    x = solver.solve(b);
    for (int idx = 0; idx < NUM_IMAGES; ++idx) {
        convertVectorToMesh(x, mesh_cpu_x.at(idx), mesh_cpu_y.at(idx), idx);
    }
    if (VISUALIZE_WARPED) {
        for (int idx = 0; idx < full_imgs.size(); ++idx) {
            Mat visualized_mesh = drawMesh(mesh_cpu_x[idx], mesh_cpu_y[idx], mesh_size[idx]);
            imshow(std::to_string(idx), visualized_mesh);
            full_imgs.lock();
            waitKey();
            full_imgs.unlock();
        }
    }

    //update previously selected features for temporal terms and dynamic calibration
    prev_features = features;
    for (int idx = 0; idx < NUM_IMAGES; ++idx) {
        prev_matches[idx] = selected_matches[idx];

        int idx2 = (idx - 1 == -1) ? NUM_IMAGES - 1 : idx-1;
        //TODO: don't skip matched features
        if (use_old_features[idx] && !prev_matches[idx].empty() && !old_features.at(idx).empty()) {
            continue;
        }


        old_matches[idx] = selected_matches[idx];
        old_features.at(idx) = features;
        prev_avg[idx * 2] = fp_avg[idx * 2];
        prev_avg[idx * 2 + 1] = fp_avg[idx * 2 + 1];

        old_matches[idx2] = selected_matches[idx2];
        old_features.at(idx2) = features;
        prev_avg[idx2 * 2] = fp_avg[idx2 * 2];
        prev_avg[idx2 * 2 + 1] = fp_avg[idx2 * 2 + 1];

    }
}

std::vector<cv::Mat> MeshWarper::interpolateMesh(vector<Mat> &mesh_start, vector<Mat> &mesh_end, float progress)
{
    int mesh_count = mesh_start.size();
    std::vector<cv::Mat> interp_mesh(mesh_count);
    for (int idx = 0; idx < mesh_count; ++idx) {
        int rows = mesh_start.at(idx).rows;
        int cols = mesh_start.at(idx).cols;
        interp_mesh.at(idx) = Mat(N, M, CV_32FC1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float start = mesh_start.at(idx).at<float>(i, j);
                float end = mesh_end.at(idx).at<float>(i, j);
                interp_mesh.at(idx).at<float>(i, j) = (start + (end - start) * progress);
            }
        }
    }
    return interp_mesh;
}

void MeshWarper::calibrateMeshWarp(LockableVector<Mat> &full_imgs, vector<ImageFeatures> &features,
                       vector<MatchesInfo> &pairwise_matches, LockableVector<cuda::GpuMat> &x_mesh,
                       LockableVector<cuda::GpuMat> &y_mesh, vector<cuda::GpuMat> &x_maps,
                       vector<cuda::GpuMat> &y_maps) {
    full_imgs.lock();
    int full_imgs_count = full_imgs.size();
    full_imgs.unlock();
    vector<Mat> simple_mesh_x(full_imgs_count);
    vector<Mat> simple_mesh_y(full_imgs_count);
    vector<Size> mesh_size(full_imgs_count);
    createMesh(full_imgs, features, pairwise_matches, simple_mesh_x, simple_mesh_y, x_maps, y_maps, mesh_size);

    convertMeshesToMap(simple_mesh_x, simple_mesh_y, x_mesh, y_mesh, mesh_size);
    if (VISUALIZE_WARPED) {
        for (int idx = 0; idx < full_imgs.size(); ++idx) {
            Mat visualized_mesh = drawMesh(simple_mesh_x[idx], simple_mesh_y[idx], mesh_size[idx]);
            imshow(std::to_string(idx), visualized_mesh);
            waitKey();
        }
    }
}

void MeshWarper::recalibrateMesh(LockableVector<cv::Mat> &full_img, std::vector<cv::cuda::GpuMat> &x_maps,
                     std::vector<cv::cuda::GpuMat> &y_maps, LockableVector<cv::cuda::GpuMat> &x_mesh,
                     LockableVector<cv::cuda::GpuMat> &y_mesh)
{
    vector<ImageFeatures> features(NUM_IMAGES);
    vector<MatchesInfo> pairwise_matches(NUM_IMAGES - 1 + (int)wrapAround);

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
            for (int t = 0; t < 8; t += 1) {

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

                // Calculated from equation [6] http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
                // Vn in paper is in code [xn, yn]
                float u = (-V1x*V2y+V1x*V3y-V2x*V1y+2*V2x*V2y-V2x*V3y+V3x*V1y-V3x*V2y)/(2*(V2x-V3x)*(V2y-V3y));
                float v = (V1x*V2y-V1x*V3y-V2x*V1y+V2x*V3y+V3x*V1y-V3x*V2y)/(2*(V2x-V3x)*(V2y-V3y));
                float cell_width = (width / (M - 1));
                float cell_height = (height / (N - 1));


                Point2i Vi_rel[3] = Vi;

                int min_v_x = min(min(Vi[0].x, Vi[1].x), Vi[2].x);
                int min_v_y = min(min(Vi[0].y, Vi[1].y), Vi[2].y);

                if (min_v_x < 0) {
                    for (int v = 0; v < 3; ++v) {
                        Vi_rel[v].x++;
                    }
                }
                if (min_v_y < 0) {
                    for (int v = 0; v < 3; ++v) {
                        Vi_rel[v].y++;
                    }
                }

                float sal; // Salience of the triangle. Using 0.5f + l2-norm of color values of the triangle
                Mat mask(cell_height, cell_width, CV_8UC1);
                mask.setTo(Scalar::all(0));
                Point pts[3];
                pts[0] = Point(Vi_rel[0].x * cell_width, Vi_rel[0].y * cell_height);
                pts[1] = Point(Vi_rel[1].x * cell_width, Vi_rel[1].y * cell_height);
                pts[2] = Point(Vi_rel[2].x * cell_width, Vi_rel[2].y * cell_height);
                fillConvexPoly(mask, pts, 3, Scalar(255));

                Mat mean;
                Mat deviation;

                Rect crop;
                crop.x = min(min(V1x, V2x), V3x);
                crop.y = min(min(V1y, V2y), V3y);
                crop.width = cell_width;
                crop.height = cell_height;

                meanStdDev(image(crop), mean, deviation, mask);
                Mat variance;
                pow(deviation, 2, variance);
                sal = sqrt(norm(variance, NORM_L2) + 0.5f);



                A.insert(row,   2*((Vi_total[0].x) + M * (Vi_total[0].y) + M*N*idx)) = a * sal;
                A.insert(row,   2*((Vi_total[0].x) + M * (Vi_total[0].y) + M*N*idx) + 1) = a * sal;
                A.insert(row,   2*((Vi_total[1].x) + M * (Vi_total[1].y) + M*N*idx)) = a * (u - v - 1) * sal;
                A.insert(row,   2*((Vi_total[1].x) + M * (Vi_total[1].y) + M*N*idx) + 1) = a * (u + v - 1) * sal;
                A.insert(row,   2*((Vi_total[2].x) + M * (Vi_total[2].y) + M*N*idx)) = a * (-u + v) * sal;
                A.insert(row,   2*((Vi_total[2].x) + M * (Vi_total[2].y) + M*N*idx) + 1) = a * (-u - v) * sal;

                // b should be zero anyway, but for posterity's sake set it to zero
                b(row) = 0;


                A.insert(row + 1,   2*((Vi_total[0].x) + M * (Vi_total[0].y) + M*N*idx)) = a * sal;
                A.insert(row + 1,   2*((Vi_total[0].x) + M * (Vi_total[0].y) + M*N*idx) + 1) = a * sal;
                A.insert(row + 1,   2*((Vi_total[1].x) + M * (Vi_total[1].y) + M*N*idx)) = a * (u - v - 1) * sal;
                A.insert(row + 1,   2*((Vi_total[1].x) + M * (Vi_total[1].y) + M*N*idx) + 1) = a * (u + v - 1) * sal;
                A.insert(row + 1,   2*((Vi_total[2].x) + M * (Vi_total[2].y) + M*N*idx)) = a * (-u + v) * sal;
                A.insert(row + 1,   2*((Vi_total[2].x) + M * (Vi_total[2].y) + M*N*idx) + 1) = a * (-u - v) * sal;

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

void MeshWarper::calcTemporalLocalTerm(int &row, const vector<matchWithDst_t> &matches,
                                       const vector<ImageFeatures> &features, int idx)
{
    float f = focal_length;
    float a = sqrt(ALPHAS[3]);
    for (int m = 0; m < matches.size(); ++m) {
        const matchWithDst_t matchWDst = matches.at(m);
        DMatch match = matches.at(m).match;

        int queryIdx = match.queryIdx;
        int trainIdx = match.trainIdx;
        int src = idx;
        int dst = idx;

        float h1 = features[src].img_size.height;
        float w1 = features[src].img_size.width;
        float h2 = prev_features[dst].img_size.height;
        float w2 = prev_features[dst].img_size.width;

        Point2f p1 = features[src].keypoints[queryIdx].pt;
        Point2f p2 = prev_features[dst].keypoints[trainIdx].pt;

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


        // fp1 bilinear mapping
        // from: https://www2.eecs.berkeley.edu/Pubs/TechRpts/1989/CSD-89-516.pdf
        A.insert(row,  2*(l1   + M * (t1)   + M*N*src)) = (1-u1)*(1-v1) * a;
        A.insert(row,  2*(l1+1 + M * (t1)   + M*N*src)) = u1*(1-v1) * a;
        A.insert(row,  2*(l1 +   M * (t1+1) + M*N*src)) = v1*(1-u1) * a;
        A.insert(row,  2*(l1+1 + M * (t1+1) + M*N*src)) = u1*v1 * a;
        b(row) = p2.x * a;


        // fp1 bilinear mapping
        A.insert(row+1, 2*(l1   + M * (t1)   + M*N*src)+1) = (1-u1)*(1-v1) * a;
        A.insert(row+1, 2*(l1+1 + M * (t1)   + M*N*src)+1) = u1*(1-v1) * a;
        A.insert(row+1, 2*(l1 +   M * (t1+1) + M*N*src)+1) = v1*(1-u1) * a;
        A.insert(row+1, 2*(l1+1 + M * (t1+1) + M*N*src)+1) = u1*v1 * a;
        b(row + 1) = p2.y * a;

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

// TODO: convert here normal mesh size to opencv mesh size
// Convert the mesh into a backward map used by opencv remap function
// @TODO implement this in CUDA so it can be run entirely on GPU
void MeshWarper::convertMeshesToMap(vector<cv::Mat> &mesh_cpu_x, vector<cv::Mat> &mesh_cpu_y,
                        LockableVector<cv::cuda::GpuMat> &map_x,
                        LockableVector<cv::cuda::GpuMat> &map_y, vector<cv::Size> mesh_sizes)
{
    for (int idx = 0; idx < mesh_sizes.size(); ++idx) {
        Size mesh_size = mesh_sizes.at(idx);
        cuda::GpuMat gpu_small_mesh_x;
        cuda::GpuMat gpu_small_mesh_y;
        cuda::GpuMat big_x;
        cuda::GpuMat big_y;
        Mat big_mesh_x;
        Mat big_mesh_y;
        // Interpolate pixel positions between the mesh vertices with a custom resize function
        gpu_small_mesh_x.upload(mesh_cpu_x.at(idx));
        gpu_small_mesh_y.upload(mesh_cpu_y.at(idx));
        custom_resize(gpu_small_mesh_x, big_x, mesh_size);
        custom_resize(gpu_small_mesh_y, big_y, mesh_size);
        big_x.download(big_mesh_x);
        big_y.download(big_mesh_y);

        // Calculate pixel values for a map half the width and height and then resize the map back to
        // full size
        int scale = 2;
        Mat warp_x(mesh_size.height / scale, mesh_size.width / scale, mesh_cpu_x.at(idx).type());
        Mat warp_y(mesh_size.height / scale, mesh_size.width / scale, mesh_cpu_y.at(idx).type());
        Mat set_values(mesh_size.height / scale, mesh_size.width / scale, CV_32F);
        set_values.setTo(Scalar::all(0));
        warp_x.setTo(Scalar::all(0));
        warp_y.setTo(Scalar::all(0));

        Mat sum_x(mesh_size.height / scale, mesh_size.width / scale, mesh_cpu_x.at(idx).type());
        Mat sum_y(mesh_size.height / scale, mesh_size.width / scale, mesh_cpu_y.at(idx).type());
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
        map_x.lock();
        custom_resize(gpu_small_mesh_x, map_x.at(idx), mesh_size);
        map_x.unlock();
        map_y.lock();
        custom_resize(gpu_small_mesh_y, map_y.at(idx), mesh_size);
        map_y.unlock();
    }
}

void MeshWarper::filterMatches(vector<MatchesInfo> &pairwise_matches, vector<ImageFeatures> &features, vector<matchWithDst_t> *filt_matches)
{
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

        //Find all indexes of the inliers_mask that contain the value 1
        for (int i = 0; i < pw_matches.inliers_mask.size(); ++i) {
            if (pw_matches.inliers_mask[i]) {

                //Filter out matches that don't seem sane for the current rig
                matchWithDst_t match = { pw_matches.matches[i], pw_matches.dst_img_idx };
                int queryIdx = match.match.queryIdx;
                int trainIdx = match.match.trainIdx;
                int src = pw_matches.src_img_idx;
                int dst = pw_matches.dst_img_idx;

                Point2f p1 = features[src].keypoints[queryIdx].pt;
                Point2f p2 = features[dst].keypoints[trainIdx].pt;

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

                float scale = compose_scale / work_scale;
                float max_x_dist = theta * focal_length * scale;

                if (abs(p1.y - p2.y) > 40) {
                    //LOGLN("Invalid featurepoint match y-diff: " << p1.y - p2.y);
                    continue;
                } else if (abs(max_x_dist - (p1.x - p2.x)) > 300) {
                    //LOGLN("Invalid featurepoint match x-diff: " << abs(max_x_dist - (p1.x - p2.x)));
                    continue;
                }
                filt_matches[src].push_back(match);
            }
        }
    }
}

void MeshWarper::filterTemporalMatches(vector<MatchesInfo> &pairwise_matches, vector<ImageFeatures> &features, vector<matchWithDst_t> *filt_matches)
{
    // Select all matches that fit criteria
    for (int idx = 0; idx < pairwise_matches.size(); ++idx) {
        MatchesInfo &pw_matches = pairwise_matches[idx];
        int src = pw_matches.src_img_idx;
        int dst = pw_matches.dst_img_idx;
        if (!pw_matches.matches.size() || !pw_matches.num_inliers) continue;

        //Find all indexes of the inliers_mask that contain the value 1
        for (int i = 0; i < pw_matches.inliers_mask.size(); ++i) {
            if (pw_matches.inliers_mask[i]) {

                //Filter out matches that don't seem sane for the current rig
                matchWithDst_t match = { pw_matches.matches[i], pw_matches.dst_img_idx };
                int queryIdx = match.match.queryIdx;
                int trainIdx = match.match.trainIdx;
                int src = pw_matches.src_img_idx;
                int dst = pw_matches.dst_img_idx;

                Point2f p1 = features[src].keypoints[queryIdx].pt;
                Point2f p2 = prev_features[dst].keypoints[trainIdx].pt;

                if (abs(p1.y - p2.y) > 30) {
                    //LOGLN("Invalid TEMPORAL featurepoint match y-diff: " << p1.y - p2.y);
                    continue;
                } else if (abs(p1.x - p2.x) > 30) {
                    //LOGLN("Invalid TEMPORAL featurepoint match x-diff: " << p1.x - p2.x);
                    continue;
                }
                filt_matches[src].push_back(match);
            }
        }
    }
}
