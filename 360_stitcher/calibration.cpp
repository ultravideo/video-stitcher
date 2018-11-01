#include "calibration.h"
#include <algorithm>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaarithm.hpp"

#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"

using namespace cv;
using namespace cv::detail;
using std::vector;
 

void findFeatures(vector<Mat> &full_img, vector<ImageFeatures> &features,
                  const double &work_scale) {
    Ptr<cuda::ORB> d_orb = cuda::ORB::create(1500, 1.2f, 8);
    Ptr<SurfFeaturesFinderGpu> surf = makePtr<SurfFeaturesFinderGpu>(HESS_THRESH, NOCTAVES, NOCTAVESLAYERS);
    Mat image;
    cuda::GpuMat gpu_img;
    cuda::GpuMat descriptors;
    // Read images from file and resize if necessary
    for (int i = 0; i < NUM_IMAGES; i++) {
        // Negative value means processing images in the original size
        if (WORK_MEGAPIX < 0)
        {
            image = full_img[i];
        }
        // Else downscale images to speed up the process
        else
        {
            cv::resize(full_img[i], image, Size(), work_scale, work_scale);
        }

        if (use_surf) {
            // Find features with SURF feature finder
            (*surf)(image, features[i]);
        }
        else
        {
            // Find features with ORB feature finder
            gpu_img.upload(image);
            cuda::cvtColor(gpu_img, gpu_img, CV_BGR2GRAY);
            features[i].img_size = image.size();
            d_orb->detectAndCompute(gpu_img, noArray(), features[i].keypoints, descriptors);
            descriptors.download(features[i].descriptors);
        }
        features[i].img_idx = i;
    }
}


void matchFeatures(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches) {
    // Use different way of matching features for SURF and ORB
    if (use_surf) {
        Ptr<FeaturesMatcher> fm = makePtr<BestOf2NearestMatcher>(true, MATCH_CONF);
        (*fm)(features, pairwise_matches);
        return;
    }
    Ptr<DescriptorMatcher> dm = DescriptorMatcher::create("BruteForce-Hamming");

    // Match features
    for (int i = 0; i < pairwise_matches.size(); ++i) {
        int idx1 = (i + 1 == NUM_IMAGES) ? i-1 : i;
        int idx2 = (i + 1 == NUM_IMAGES) ? 0 : i+1;
        pairwise_matches[i].src_img_idx = idx1;
        pairwise_matches[i].dst_img_idx = idx2;
        ImageFeatures f1 = features[idx1];
        ImageFeatures f2 = features[idx2];
        vector<vector<DMatch>> matches;
        dm->knnMatch(f1.descriptors, f2.descriptors, matches, 2);
        for (int j = 0; j < matches.size(); ++j) {
            if (matches[j][0].distance < 0.7 * matches[j][1].distance) {
                pairwise_matches[i].matches.push_back(matches[j][0]);
            }
        }
        Mat src_points(1, static_cast<int>(pairwise_matches[i].matches.size()), CV_32FC2);
        Mat dst_points(1, static_cast<int>(pairwise_matches[i].matches.size()), CV_32FC2);
        if (pairwise_matches[i].matches.size()) {
            for (int j = 0; j < pairwise_matches[i].matches.size(); ++j) {
                const DMatch& m = pairwise_matches[i].matches[j];

                Point2f p = f1.keypoints[m.queryIdx].pt;
                p.x -= f1.img_size.width * 0.5f;
                p.y -= f1.img_size.height * 0.5f;
                src_points.at<Point2f>(0, static_cast<int>(j)) = p;

                p = f2.keypoints[m.trainIdx].pt;
                p.x -= f2.img_size.width * 0.5f;
                p.y -= f2.img_size.height * 0.5f;
                dst_points.at<Point2f>(0, static_cast<int>(j)) = p;
            }

            // Find pair-wise motion
            pairwise_matches[i].H = findHomography(src_points, dst_points, pairwise_matches[i].inliers_mask, RANSAC);

            // Find number of inliers
            pairwise_matches[i].num_inliers = 0;
            for (size_t idx = 0; idx < pairwise_matches[i].inliers_mask.size(); ++idx)
                if (pairwise_matches[i].inliers_mask[idx])
                    pairwise_matches[i].num_inliers++;

            // Confidence calculation copied from opencv feature matching code
            // These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
            // using Invariant Features"
            pairwise_matches[i].confidence = pairwise_matches[i].num_inliers /
                                             (8 + 0.3 * pairwise_matches[i].matches.size());

            // Set zero confidence to remove matches between too close images, as they don't provide
            // additional information anyway. The threshold was set experimentally.
            pairwise_matches[i].confidence = pairwise_matches[i].confidence > 3. ?
                                             0. : pairwise_matches[i].confidence;
        }

        pairwise_matches[i].confidence = 1;
    }

}


bool calibrateCameras(vector<CameraParams> &cameras, const cv::Size full_img_size,
                      const double work_scale) {
    cameras = vector<CameraParams>(NUM_IMAGES);
	double fov = 90.0*PI/180.0;
	double focal_tmp = 1.0/(tan(fov*0.5));

    for (int i = 0; i < cameras.size(); ++i) {
        float rot = static_cast<float>(2.0 * PI * static_cast<float>(i) / NUM_IMAGES); //kameroiden paikat ympyrän kehällä.

		Mat Rx = (Mat_<float>(3, 3) <<
			1, 0, 0,
			0, cos(0), -sin(0),
			0, sin(0), cos(0)); //Only for example as we do not have this rotation the matrix becomes an identity matrix --> no need

		Mat Ry = (Mat_<float>(3, 3) <<
			cos(rot), 0, sin(rot),
			0, 1, 0,
			-sin(rot), 0, cos(rot)); //The cameras in the grid are rotated around y-axis. No other rotation is present.

		Mat Rz = (Mat_<float>(3, 3) <<
			cos(0), -sin(0), 0,
			sin(0), cos(0), 0,
			0, 0, 1); //Only for example as we do not have this rotation the matrix becomes an identity matrix --> no need

		Mat rotMat = Rz * Ry * Rx; //the order to combine euler angle matrixes to rotation matrix is ZYX!

        cameras[i].R = rotMat;
		cameras[i].ppx = (full_img_size.width * work_scale) / 2.0;
        cameras[i].ppy = (full_img_size.height * work_scale) / 2.0; // principal point y
		cameras[i].aspect = 16.0 / 9.0; //as it is known the cameras have 1080p resolution, the aspect ratio is known to be 16:9
		//in 1080p resolution with medium fov (127 degrees) the focal lengt is 21mm.
		// with fov 170 degrees the focal lenth is 14mm and with fov 90 degrees the focal length is 28mm 
		// This information from gopro cameras is found from internet may be different for the model used in the grid!!!
		
		cameras[i].focal = focal_tmp * cameras[i].ppx;
		std::cout << "Focal " << i << ": " << cameras[i].focal << std::endl;
    }

    return true;
}


// Precalculates warp maps for images so only precalculated maps are used to warp online
void warpImages(vector<Mat> full_img, Size full_img_size, vector<CameraParams> cameras,
                Ptr<Blender> blender, Ptr<ExposureCompensator> compensator, double work_scale,
                double seam_scale, double seam_work_aspect, vector<cuda::GpuMat> &x_maps,
                vector<cuda::GpuMat> &y_maps, double &compose_scale, float &warped_image_scale,
                float &blend_width) {
    // STEP 3: warping images // ------------------------------------------------------------------------------------------------
    cuda::Stream stream;
    vector<cuda::GpuMat> gpu_imgs(NUM_IMAGES);
    vector<cuda::GpuMat> gpu_seam_imgs(NUM_IMAGES);
    vector<cuda::GpuMat> gpu_seam_imgs_warped(NUM_IMAGES);
    vector<cuda::GpuMat> gpu_masks(NUM_IMAGES);
    vector<cuda::GpuMat> gpu_masks_warped(NUM_IMAGES);
    vector<UMat> masks_warped(NUM_IMAGES);
    vector<UMat> images_warped(NUM_IMAGES);
    vector<UMat> masks(NUM_IMAGES);
    vector<Size> sizes(NUM_IMAGES);
    vector<Mat> images(NUM_IMAGES);

    // Create masks for warping
#pragma omp parallel for
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        gpu_imgs[i].upload(full_img[i], stream);
        cuda::resize(gpu_imgs[i], gpu_seam_imgs[i], Size(), seam_scale, seam_scale, 1, stream);
        gpu_masks[i].create(gpu_seam_imgs[i].size(), CV_8U);
        gpu_masks[i].setTo(Scalar::all(255), stream);
    }

    Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarperGpu>();
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
    cv::detail::CylindricalWarperGpu* gpu_warper = dynamic_cast<cv::detail::CylindricalWarperGpu*>(warper.get());

    vector<UMat> images_warped_f(NUM_IMAGES);
    vector<Point> corners(NUM_IMAGES);

    // Warp image
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0, 0) *= swa;
        K(0, 2) *= swa;
        K(1, 1) *= swa;
        K(1, 2) *= swa;

        corners[i] = gpu_warper->warp(gpu_seam_imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, gpu_seam_imgs_warped[i]);
        sizes[i] = gpu_seam_imgs_warped[i].size();
        gpu_seam_imgs_warped[i].download(images_warped[i], stream);

        gpu_warper->warp(gpu_masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, gpu_masks_warped[i]);
        gpu_masks_warped[i].download(masks_warped[i]);

        gpu_seam_imgs_warped[i].convertTo(gpu_seam_imgs_warped[i], CV_32F, stream);
        gpu_seam_imgs_warped[i].download(images_warped_f[i], stream);
    }

    // STEP 4: compensating exposure and finding seams // -----------------------------------------------------------------------

    compensator->feed(corners, images_warped, masks_warped);
    GainCompensator* gain_comp = dynamic_cast<GainCompensator*>(compensator.get());

    Ptr<SeamFinder> seam_finder = makePtr<VoronoiSeamFinder>();
    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    //  STEP 5: composing panorama // -------------------------------------------------------------------------------------------
    double compose_work_aspect = 1;

    // Negative value means compose in original resolution
    if (COMPOSE_MEGAPIX > 0)
    {
        compose_scale = min(1.0, sqrt(COMPOSE_MEGAPIX * 1e6 / full_img[0].size().area()));
    }

    // Compute relative scale
    compose_work_aspect = compose_scale / work_scale;

    // Update warped image scale
    warper = warper_creator->create(warped_image_scale * static_cast<float>(compose_work_aspect));
    gpu_warper = dynamic_cast<cv::detail::CylindricalWarperGpu*>(warper.get());

    Size sz = full_img_size;

    if (std::abs(compose_scale - 1) > 1e-1)
    {
        sz.width = cvRound(full_img_size.width * compose_scale);
        sz.height = cvRound(full_img_size.height * compose_scale);
    }

    // Update corners and sizes
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        // Update intrinsics
        cameras[i].focal *= compose_work_aspect;
        cameras[i].ppx *= compose_work_aspect;
        cameras[i].ppy *= compose_work_aspect;

        // Update corners and sizes
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = warper->warpRoi(sz, K, cameras[i].R);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
    }

    Size dst_sz = resultRoi(corners, sizes).size();
    blend_width = sqrt(static_cast<float>(dst_sz.area())) * BLEND_STRENGTH / 100.f;

    if (blend_width < 1.f)
    {
        blender = Blender::createDefault(Blender::NO, true);
    }
    else
    {
        MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
        mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
    }

    blender->prepare(corners, sizes);


    MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
    int64 start = getTickCount();
    Size img_size;
    cuda::GpuMat gpu_img;
    cuda::GpuMat gpu_mask;
    cuda::GpuMat gpu_mask_warped;
    cuda::GpuMat gpu_seam_mask;
    Mat mask_warped;
    Ptr<cuda::Filter> dilation_filter = cuda::createMorphologyFilter(MORPH_DILATE, CV_8U, Mat());
    for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
    {
        gpu_mask_warped.release();
        img_size = full_img[img_idx].size();
        img_size = Size((int)(img_size.width * compose_scale), (int)(img_size.height * compose_scale));
        full_img[img_idx].release();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Create warping map for online process
        gpu_warper->buildMaps(img_size, K, cameras[img_idx].R, x_maps[img_idx], y_maps[img_idx]);

        // Warp the current image mask
        gpu_mask.create(img_size, CV_8U);
        gpu_mask.setTo(Scalar::all(255));

        gpu_warper->warp(gpu_mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, gpu_mask_warped);

        gpu_masks_warped[img_idx].upload(masks_warped[img_idx]);

        dilation_filter->apply(gpu_masks_warped[img_idx], gpu_seam_mask);
        cuda::resize(gpu_seam_mask, gpu_seam_mask, gpu_mask_warped.size(), 0.0, 0.0, 1);
        cuda::bitwise_and(gpu_seam_mask, gpu_mask_warped, gpu_mask_warped, noArray());

        // Calculate mask pyramid for online process
        mb->init_gpu(gpu_img, gpu_mask_warped, corners[img_idx]);
    }

    Mat result;
    Mat result_mask;

    blender->blend(result, result_mask);

}


void calibrateMeshWarp(vector<Mat> &full_imgs, vector<ImageFeatures> features,
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

    int features_per_image = MAX_FEATURES_PER_IMAGE;
    // 2 * images.size()*N*M for global alignment, 2*images.size()*(N-1)*(M-1) for smoothness term and
    // 2 * (NUM_IMAGES +(int)wrapAround) * features_per_image for local alignment
    int num_rows = 2 * static_cast<int>(images.size())*N*M + 2* static_cast<int>(images.size())*(N-1)*(M-1) +
                   2 * (NUM_IMAGES + (int)wrapAround) * features_per_image;
    Eigen::SparseMatrix<double> A(num_rows, 2*N*M*images.size());
    Eigen::VectorXd b(num_rows);
    Eigen::VectorXd x;
    b.fill(0);

    int global_start = 0;
    int smooth_start = 2 * static_cast<int>(images.size())*N*M;
    int local_start = 2 * static_cast<int>(images.size())*N*M + 2* static_cast<int>(images.size())*(N-1)*(M-1);

    // Global alignment term from http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
    float a = ALPHAS[1];
    int row = 0;
    for (int idx = 0; idx < images.size(); ++idx) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                float x1 = static_cast<float>(j * mesh_size[idx].width / (M-1));
                float y1 = static_cast<float>(i * mesh_size[idx].height / (N-1));
                float scale = compose_scale / work_scale;
                float tau = 1;
                for (int ft = 0; ft < features[idx].keypoints.size(); ++ft) {
                    Point ft_point = features[idx].keypoints[ft].pt;
                    if (sqrt(pow(ft_point.x * scale - x1, 2) + pow(ft_point.y * scale - y1, 2)) < GLOBAL_DIST) {
                        tau = 0;
                        break;
                    }
                }
                A.insert(row, 2*(j + i*M + idx*M*N)) = a * tau;
                A.insert(row + 1, 2*(j + i*M + idx*M*N)+1) = a * tau;
                b(2*(j + i*M + idx*M*N)) = a * tau * x1;
                b(2*(j + i*M + idx*M*N)+1) = a * tau * y1;
                row += 2;
            }
        }
    }

    row = 0;
    // Smoothness term from http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
    a = ALPHAS[2];
    for (int idx = 0; idx < images.size(); ++idx) {
        for (int i = 0; i < N-1; ++i) {
            for (int j = 0; j < M-1; ++j) {
                float x1 = mesh_cpu_x[idx].at<float>(i, j);
                float x2 = mesh_cpu_x[idx].at<float>(i+1, j);
                float x3 = mesh_cpu_x[idx].at<float>(i, j+1);
                float x4 = mesh_cpu_x[idx].at<float>(i+1, j+1);
                float y1 = mesh_cpu_y[idx].at<float>(i, j);
                float y2 = mesh_cpu_y[idx].at<float>(i+1, j);
                float y3 = mesh_cpu_y[idx].at<float>(i, j+1);
                float y4 = mesh_cpu_y[idx].at<float>(i+1, j+1);
                float dx = x3 - x2;
                float dy = y3 - y2;
                float dx2 = x4 - x3;
                float dy2 = y4 - y3;
                float u = (dx*x1-dx*x2+dy*y1-dy*y2)/(dx*dx+dy*dy);
                float v = (-dx*y1+dx*y2+x1*dy-x2*dy)/(dx*dx+dy*dy);
                float u2 = (dx2*x2-dx2*x3+dy2*y2-dy2*y3)/(dx2*dx2+dy2*dy2);
                float v2 = (-dx2*y2+dx2*y3+x2*dy2-x3*dy2)/(dx2*dx2+dy2*dy2);

                float sal; // Salience of the triangle. Using 0.5f + l2-norm of color values of the triangle

                Mat mask(images[idx].rows, images[idx].cols, CV_8UC1);
                mask.setTo(Scalar::all(0));
                Point pts[3] = { Point(x1, y1), Point(x2, y1), Point(y2, x1) };
                fillConvexPoly(mask, pts, 3, Scalar(1));
                sal = 0.5f + norm(images[idx], NORM_L2, mask) / (255*255);
                
                A.insert(smooth_start + row,   2*(j + M * i + M*N*idx)) = a * sal; // x1
                A.insert(smooth_start + row,   2*(j + M * i + M*N*idx) + 1) = a * sal; // y1
                A.insert(smooth_start + row,   2*(j + M * (i+1) + M*N*idx)) = a*(u + v - 1) * sal; // x2
                A.insert(smooth_start + row,   2*(j + M * (i+1) + M*N*idx) + 1) = a*(u - v - 1) * sal; // y2
                A.insert(smooth_start + row,   2*(j+1 + M * i + M*N*idx)) = a*(-u - v) * sal; // x3
                A.insert(smooth_start + row,   2*(j+1 + M * i + M*N*idx) + 1) = a*(-u + v) * sal; // y3

                mask.setTo(Scalar::all(0));
                Point pts2[3] = { Point(x2, y1), Point(x1, y2), Point(y2, x2) };
                fillConvexPoly(mask, pts2, 3, Scalar(1));
                sal = 0.5f + norm(images[idx], NORM_L2, mask) / (255*255);

                A.insert(smooth_start + row+1, 2*(j + M * (i+1) + M*N*idx)) = a * sal; // x2
                A.insert(smooth_start + row+1, 2*(j + M * (i+1) + M*N*idx) + 1) = a * sal; // y2
                A.insert(smooth_start + row+1, 2*(j+1 + M * i + M*N*idx)) = a*(u + v - 1) * sal; // x3
                A.insert(smooth_start + row+1, 2*(j+1 + M * i + M*N*idx) + 1) = a*(u - v - 1) * sal; // y3
                A.insert(smooth_start + row+1, 2*(j+1 + M * (i+1) + M*N*idx)) = a*(-u - v) * sal; // x4
                A.insert(smooth_start + row+1, 2*(j+1 + M * (i+1) + M*N*idx) + 1) = a*(-u + v) * sal; // x4
                row += 2;
            }
        }
    }

    if (VISUALIZE_MATCHES) { // Draw the meshes for visualisation
        for (int idx = 0; idx < NUM_IMAGES; ++idx) {
            for (int i = 0; i < mesh_cpu_x[idx].rows - 1; ++i) {
                for (int j = 0; j < mesh_cpu_x[idx].cols; ++j) {
                    Point start = Point(mesh_cpu_x[idx].at<float>(i, j), mesh_cpu_y[idx].at<float>(i, j));
                    Point end = Point(mesh_cpu_x[idx].at<float>(i + 1, j), mesh_cpu_y[idx].at<float>(i + 1, j));
                    line(images[idx], start, end, Scalar(255, 0, 0), 2);
                }
            }
            for (int i = 0; i < mesh_cpu_x[idx].rows; ++i) {
                for (int j = 0; j < mesh_cpu_x[idx].cols - 1; ++j) {
                    Point start = Point(mesh_cpu_x[idx].at<float>(i, j), mesh_cpu_y[idx].at<float>(i, j));
                    Point end = Point(mesh_cpu_x[idx].at<float>(i, j + 1), mesh_cpu_y[idx].at<float>(i, j + 1));
                    line(images[idx], start, end, Scalar(255, 0, 0), 2);
                }
            }
        }
    }

    // Local alignment term from http://web.cecs.pdx.edu/~fliu/papers/cvpr2014-stitching.pdf
    row = 0;
    float f = focal_length;
    a = ALPHAS[0];
	vector<int> valid_indexes_orig;
	vector<int> valid_indexes;
    for (int idx = 0; idx < pairwise_matches.size(); ++idx) {
        MatchesInfo &pw_matches = pairwise_matches[idx];
        if (!pw_matches.matches.size() || !pw_matches.num_inliers) continue;
        int src = pw_matches.src_img_idx;
        int dst = pw_matches.dst_img_idx;
        // Only calculate loss from pairs of src and dst where src = dst - 1
        // to avoid using same pairs multiple times
        if (abs(src - dst - 1) > 0.1) { 
            continue;
        }

		valid_indexes_orig.clear();
		//Find all indexes of the inliers_mask that contain the value 1
		for (int i = 0; i < pw_matches.inliers_mask.size(); i++) {
			if (pw_matches.inliers_mask[i]) {
				valid_indexes_orig.push_back(i);
			}
		}
		if (valid_indexes_orig.size() == 0) continue;

        for (int i = 0; i < min(features_per_image, (int)(pw_matches.matches.size() * 0.8f)); ++i) {
			//Shuffle the index vector on each loop to get random results each time
			valid_indexes = valid_indexes_orig;
			std::random_shuffle(valid_indexes.begin(), valid_indexes.end());
			while (1) {
				if (valid_indexes.size() == 0) break;
				int idx = valid_indexes.back();
				valid_indexes.pop_back();
                // Only use inlier matches
                if (!pw_matches.inliers_mask[idx]) continue;

                int idx1 = pw_matches.matches[idx].queryIdx;
                int idx2 = pw_matches.matches[idx].trainIdx;
                KeyPoint k1 = features[src].keypoints[idx1];
                KeyPoint k2 = features[dst].keypoints[idx2];

                float h1 = features[src].img_size.height;
                float w1 = features[src].img_size.width;
                float h2 = features[dst].img_size.height;
                float w2 = features[dst].img_size.width;

                float x1 = k1.pt.x - w1 / 2;
                float y1 = k1.pt.y - h1 / 2;
                float x2 = k2.pt.x - w2 / 2;
                float y2 = k2.pt.y - h2 / 2;

                float theta = dst - src;
                if (src == 0 && dst == NUM_IMAGES - 1 && wrapAround) {
                    theta = -1;
                }
                theta *= 2 * PI / 6;

                // Feature points are from work scale images change scale to compose scale
                float scale = compose_scale / work_scale;
                float x1_ = (f * atan(x1 / f) + w1 / 2) * scale;
                float x2_ = (f * atan(x2 / f) + w2 / 2) * scale;
                float y1_ = (f * y1 / sqrt(x1*x1 + f*f) + h1 / 2) * scale;
                float y2_ = (f * y2 / sqrt(x2*x2 + f*f) + h2 / 2) * scale;

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
                    circle(images[src], Point(x1_, y1_), 3, Scalar(0, 255, 0), 3);
                    circle(images[src], Point(x2_ + f * scale * theta, y2_), 3, Scalar(0, 0, 255), 3);
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
                A.insert(local_start + row,  2*(l1   + M * (t1)   + M*N*src)) = (1-u1)*(1-v1) * a;
                A.insert(local_start + row,  2*(l1+1 + M * (t1)   + M*N*src)) = u1*(1-v1) * a;
                A.insert(local_start + row,  2*(l1 +   M * (t1+1) + M*N*src)) = v1*(1-u1) * a;
                A.insert(local_start + row,  2*(l1+1 + M * (t1+1) + M*N*src)) = u1*v1 * a;
                A.insert(local_start + row,  2*(l2   + M * (t2)   + M*N*dst)) = -(1-u2)*(1-v2) * a;
                A.insert(local_start + row,  2*(l2+1 + M * (t2)   + M*N*dst)) = -u2*(1-v2) * a;
                A.insert(local_start + row,  2*(l2 +   M * (t2+1) + M*N*dst)) = -v2*(1-u2) * a;
                A.insert(local_start + row,  2*(l2+1 + M * (t2+1) + M*N*dst)) = -u2*v2 * a;

                b(local_start + row) = theta * f * scale * a;

                // _y_ - _y2_ = 0
                A.insert(local_start + row+1, 2*(l1   + M * (t1)   + M*N*src)+1) = (1-u1)*(1-v1) * a;
                A.insert(local_start + row+1, 2*(l1+1 + M * (t1)   + M*N*src)+1) = u1*(1-v1) * a;
                A.insert(local_start + row+1, 2*(l1 +   M * (t1+1) + M*N*src)+1) = v1*(1-u1) * a;
                A.insert(local_start + row+1, 2*(l1+1 + M * (t1+1) + M*N*src)+1) = u1*v1 * a;
                A.insert(local_start + row+1, 2*(l2   + M * (t2)   + M*N*dst)+1) = -(1-u2)*(1-v2) * a;
                A.insert(local_start + row+1, 2*(l2+1 + M * (t2)   + M*N*dst)+1) = -u2*(1-v2) * a;
                A.insert(local_start + row+1, 2*(l2 +   M * (t2+1) + M*N*dst)+1) = -v2*(1-u2) * a;
                A.insert(local_start + row+1, 2*(l2+1 + M * (t2+1) + M*N*dst)+1) = -u2*v2 * a;
                row+=2;
                break;
            }
        }
    }

    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    x = solver.solve(b);

    cuda::GpuMat gpu_small_mesh_x;
    cuda::GpuMat gpu_small_mesh_y;
    cuda::GpuMat big_x;
    cuda::GpuMat big_y;
    Mat big_mesh_x;
    Mat big_mesh_y;
    // Convert the mesh into a backward map used by opencv remap function
    // @TODO implement this in CUDA so it can be run entirely on GPU
    for (int idx = 0; idx < full_imgs.size(); ++idx) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                mesh_cpu_x[idx].at<float>(i, j) = x(2 * (j + i * M + idx*M*N));
                mesh_cpu_y[idx].at<float>(i, j) = x(2 * (j + i*M + idx*M*N) + 1);
            }
        }
        // Interpolate pixel positions between the mesh vertices by using a custom resize function
        gpu_small_mesh_x.upload(mesh_cpu_x[idx]);
        gpu_small_mesh_y.upload(mesh_cpu_y[idx]);
        custom_resize(gpu_small_mesh_x, big_x, mesh_size[idx]);
        custom_resize(gpu_small_mesh_y, big_y, mesh_size[idx]);
        big_x.download(big_mesh_x);
        big_y.download(big_mesh_y);

        // Calculate pixel values for a map half the width and height and then resize the map back to
        // full size
        int scale = 2;
        Mat warp_x(mesh_size[idx].height / scale, mesh_size[idx].width / scale, mesh_cpu_x[idx].type());
        Mat warp_y(mesh_size[idx].height / scale, mesh_size[idx].width / scale, mesh_cpu_y[idx].type());
        Mat set_values(mesh_size[idx].height / scale, mesh_size[idx].width / scale, CV_32F);
        set_values.setTo(Scalar::all(0));
        warp_x.setTo(Scalar::all(0));
        warp_y.setTo(Scalar::all(0));
        for (int i = 0; i < mesh_size[idx].height; ++i) {
            for (int j = 0; j < mesh_size[idx].width; ++j) {
                int x = (int)big_mesh_x.at<float>(i, j) / scale;
                int y = (int)big_mesh_y.at<float>(i, j) / scale;
                if (x >= 0 && y >= 0 && x < warp_x.cols && y < warp_x.rows) {
                    if (!set_values.at<float>(y, x)) {
                        warp_x.at<float>(y, x) = (float)j;
                        warp_y.at<float>(y, x) = (float)i;
                        set_values.at<float>(y, x) = 1;
                    }
                }
            }
        }
        gpu_small_mesh_x.upload(warp_x);
        gpu_small_mesh_y.upload(warp_y);
        custom_resize(gpu_small_mesh_x, x_mesh[idx], mesh_size[idx]);
        custom_resize(gpu_small_mesh_y, y_mesh[idx], mesh_size[idx]);

        if (VISUALIZE_WARPED) {
            Mat mat(mesh_size[idx].height, mesh_size[idx].width, CV_16UC3);
            for (int i = 0; i < mesh_cpu_x[idx].rows - 1; ++i) {
                for (int j = 0; j < mesh_cpu_x[idx].cols; ++j) {
                    Point start = Point(mesh_cpu_x[idx].at<float>(i, j), mesh_cpu_y[idx].at<float>(i, j));
                    Point end = Point(mesh_cpu_x[idx].at<float>(i + 1, j), mesh_cpu_y[idx].at<float>(i + 1, j));
                    line(mat, start, end, Scalar(1, 0, 0), 3);
                }
            }
            for (int i = 0; i < mesh_cpu_x[idx].rows; ++i) {
                for (int j = 0; j < mesh_cpu_x[idx].cols - 1; ++j) {
                    Point start = Point(mesh_cpu_x[idx].at<float>(i, j), mesh_cpu_y[idx].at<float>(i, j));
                    Point end = Point(mesh_cpu_x[idx].at<float>(i, j + 1), mesh_cpu_y[idx].at<float>(i, j + 1));
                    line(mat, start, end, Scalar(1, 0, 0), 3);
                }
            }
            imshow(std::to_string(idx), mat);
            waitKey();
        }
    }
}


// Takes in maps for 3D remapping, compose scale for sizing final panorama, blender and image size.
// Returns true if all the phases of calibration are successful.
bool stitch_calib(vector<Mat> full_img, vector<CameraParams> &cameras, vector<cuda::GpuMat> &x_maps,
                  vector<cuda::GpuMat> &y_maps, vector<cuda::GpuMat> &x_mesh, vector<cuda::GpuMat> &y_mesh,
                  double &work_scale, double &seam_scale, double &seam_work_aspect, double &compose_scale,
                  Ptr<Blender> &blender, Ptr<ExposureCompensator> compensator, float &warped_image_scale,
                  float &blend_width, Size &full_img_size)
{
    // STEP 1: reading images, feature finding and matching // ------------------------------------------------------------------

    for (int i = 0; i < NUM_IMAGES; ++i) {
        //CAPTURES[i].read(full_img[i]);
        if (full_img[i].empty()) {
            LOGLN("Can't read frame from camera/file nro " << i << "...");
            return false;
        }
    }
    full_img_size = full_img[0].size();

	// Negative value means processing images in the original size
	if (WORK_MEGAPIX < 0)
	{
		work_scale = 1;
	}
	// Else downscale images to speed up the process
	else
	{
		work_scale = min(1.0, sqrt(WORK_MEGAPIX * 1e6 / full_img_size.area()));
	}
	// Calculate scale for downscaling done in seam finding process
	seam_scale = min(1.0, sqrt(SEAM_MEAGPIX * 1e6 / full_img_size.area()));
	seam_work_aspect = seam_scale / work_scale;


	vector<ImageFeatures> features(NUM_IMAGES);
	vector<MatchesInfo> pairwise_matches(NUM_IMAGES - 1 + (int)wrapAround);
	if (enable_local) {
		findFeatures(full_img, features, work_scale);
		matchFeatures(features, pairwise_matches);
	}

    // STEP 2: estimating homographies // ---------------------------------------------------------------------------------------
    if (!calibrateCameras(cameras, full_img_size, work_scale)) {
        return false;
    }
	warped_image_scale = static_cast<float>(cameras[0].focal);
	std::cout << "Warped image scale: " << warped_image_scale << std::endl;


    warpImages(full_img, full_img_size, cameras, blender, compensator,
               work_scale, seam_scale, seam_work_aspect, x_maps, y_maps, compose_scale,
               warped_image_scale, blend_width);

    if (enable_local) {
        calibrateMeshWarp(full_img, features, pairwise_matches, x_mesh, y_mesh,
                          x_maps, y_maps, cameras[0].focal, compose_scale, work_scale);
        MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
        cuda::Stream stream;
        for (int i = 0; i < full_img.size(); ++i) {
            mb->update_mask(i, x_mesh[i], y_mesh[i], stream);
        }
    }
    return true;
}

void recalibrateMesh(std::vector<cv::Mat> &full_img, std::vector<cv::cuda::GpuMat> &x_maps,
                     std::vector<cv::cuda::GpuMat> &y_maps, std::vector<cv::cuda::GpuMat> &x_mesh,
                     std::vector<cv::cuda::GpuMat> &y_mesh, float focal_length, double compose_scale,
                     const double &work_scale)
{
    vector<ImageFeatures> features(NUM_IMAGES);
    findFeatures(full_img, features, work_scale);

    vector<MatchesInfo> pairwise_matches;
    matchFeatures(features, pairwise_matches);

    calibrateMeshWarp(full_img, features, pairwise_matches, x_mesh, y_mesh, x_maps, y_maps,
                      focal_length, compose_scale, work_scale);
}
