#include <iostream>
#include <string>
#include <Windows.h>
#include "blockingqueue.h"
#include "barrier.h"
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/calib3d.hpp"

#include <fstream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <vector>
#include <limits>

#define LOGLN(msg) std::cout << msg << std::endl
#define PI 3.1415926535897932384626

using std::vector;

using namespace cv;
using namespace cv::detail;

std::mutex cout_mutex;

std::string base = "static";
int skip_frames = 220;
bool wrapAround = false;
bool recalibrate = false;
bool save_video = false;
int const NUM_IMAGES = 5;
int offsets[NUM_IMAGES] = {0, 37, 72, 72, 37}; // static
//int offsets[NUM_IMAGES] = {0, 0, 0, 0, 0, 0}; // dynamic
int const INIT_FRAME_AMT = 1;
int const INIT_SKIPS = 0;
int const RECALIB_DEL = 2;
double const WORK_MEGAPIX = 0.6;	//0.6;	//-1			// Megapix parameter is scaled to the number
double const SEAM_MEAGPIX = 0.01;							// of pixels in full image and this is used
double const COMPOSE_MEGAPIX = 1.4;	//1.4;	//2.2;	//-1	// as a scaling factor when resizing images
float const MATCH_CONF = 0.5f;
float const CONF_THRESH = 0.5f;
int const BLEND_TYPE = Blender::MULTI_BAND;					// Feather blending leaves ugly seams
float const BLEND_STRENGTH = 5;
int const HESS_THRESH = 300;
int const NOCTAVES = 4;
int const NOCTAVESLAYERS = 2;

// Test material before right videos are obtained from the camera rig
vector<VideoCapture> CAPTURES;
vector<String> video_files = {base + "/0.mp4", base + "/1.mp4", base + "/2.mp4", base + "/3.mp4", base + "/4.mp4", base + "/5.mp4"};
const int TIMES = 5;
std::chrono::high_resolution_clock::time_point times[TIMES];

// Print function for parallel threads - debugging
void msg(String const message, double const value, int const thread_id)
{
	cout_mutex.lock();
	std::cout << thread_id << ": " << message << value << std::endl;
	cout_mutex.unlock();
}
void findFeatures(vector<vector<Mat>> &full_img, vector<ImageFeatures> &features,
				  double &work_scale, double &seam_scale, double &seam_work_aspect) {
	Ptr<cuda::ORB> d_orb = cuda::ORB::create(500, 1.2, 1);
	Mat image;
	cuda::GpuMat gpu_img;
	cuda::GpuMat descriptors;
	// Read images from file and resize if necessary
	for (int i = 0; i < NUM_IMAGES; i++) {
		for (int j = 0; j < INIT_FRAME_AMT; ++j) {
			// Negative value means processing images in the original size
			if (WORK_MEGAPIX < 0)
			{
				image = full_img[j][i];
				work_scale = 1;
			}
			// Else downscale images to speed up the process
			else
			{
				work_scale = min(1.0, sqrt(WORK_MEGAPIX * 1e6 / full_img[j][i].size().area()));
				cv::resize(full_img[j][i], image, Size(), work_scale, work_scale);
			}

			// Calculate scale for downscaling done in seam finding process
			seam_scale = min(1.0, sqrt(SEAM_MEAGPIX * 1e6 / full_img[j][i].size().area()));
			seam_work_aspect = seam_scale / work_scale;

			gpu_img.upload(image);
			cuda::cvtColor(gpu_img, gpu_img, CV_BGR2GRAY);
			// Find features with SURF feature finder
			if (!j) {
				features[i].img_size = image.size();
				d_orb->detectAndCompute(gpu_img, noArray(), features[i].keypoints, descriptors);
				descriptors.download(features[i].descriptors);
			} else {
				ImageFeatures ft;
				UMat cpu_descriptors;
				d_orb->detectAndCompute(gpu_img, noArray(), ft.keypoints, descriptors);
				descriptors.download(ft.descriptors);
				features[i].keypoints.insert(features[i].keypoints.end(), ft.keypoints.begin(), ft.keypoints.end());
				vconcat(features[i].descriptors, ft.descriptors, cpu_descriptors);
				features[i].descriptors = cpu_descriptors;
			}
			features[i].img_idx = i;
		}
	}
}

void matchFeatures(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches) {
	Ptr<DescriptorMatcher> dm = DescriptorMatcher::create("BruteForce-Hamming");

	// Match features
	for (int i = 0; i < pairwise_matches.size(); ++i) {
		int idx1 = (i + 1 == NUM_IMAGES) ? i-1 : i;
		int idx2 = (i + 1 == NUM_IMAGES) ? 0 : i+1;
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
		pairwise_matches[i].src_img_idx = idx1;
		pairwise_matches[i].dst_img_idx = idx2;

		// Find pair-wise motion
		pairwise_matches[i].H = findHomography(src_points, dst_points, pairwise_matches[i].inliers_mask, RANSAC);

		// Find number of inliers
		pairwise_matches[i].num_inliers = 0;
		for (size_t i = 0; i < pairwise_matches[i].inliers_mask.size(); ++i)
			if (pairwise_matches[i].inliers_mask[i])
				pairwise_matches[i].num_inliers++;

		// Confidence calculation copied from opencv feature matching code
		// These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
		// using Invariant Features"
		pairwise_matches[i].confidence = pairwise_matches[i].num_inliers / (8 + 0.3 * pairwise_matches[i].matches.size());

		// Set zero confidence to remove matches between too close images, as they don't provide
		// additional information anyway. The threshold was set experimentally.
		pairwise_matches[i].confidence = pairwise_matches[i].confidence > 3. ? 0. : pairwise_matches[i].confidence;

		pairwise_matches[i].confidence = 1;
	}

}
bool calibrateCameras(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras, float &warped_image_scale) {
	cameras = vector<CameraParams>(NUM_IMAGES);
	//estimateFocal(features, pairwise_matches, focals);
	for (int i = 0; i < cameras.size(); ++i) {
		double rot = 2 * PI * i / 6;
		Mat rotMat(3, 3, CV_32F);
		double L[3] = {cos(rot), 0, sin(rot)};
		double u[3] = {0, 1, 0};
		double s[3] = {L[1]*u[2] - L[2]*u[1], L[2]*u[0] - L[0]*u[2], L[0]*u[1] - L[1]*u[0]};
		double y[3] = {s[1]*L[2] - s[2]*L[1], s[2]*L[0] - s[0]*L[2], s[0]*L[1] - s[1]*L[0]};
		rotMat.at<float>(0, 0) = s[0];
		rotMat.at<float>(0, 1) = s[1];
		rotMat.at<float>(0, 2) = s[2];
		rotMat.at<float>(1, 0) = y[0];
		rotMat.at<float>(1, 1) = y[1];
		rotMat.at<float>(1, 2) = y[2];
		rotMat.at<float>(2, 0) = -L[0];
		rotMat.at<float>(2, 1) = -L[1];
		rotMat.at<float>(2, 2) = -L[2];
		cameras[i].R = rotMat;
		cameras[i].ppx = features[i].img_size.width / 2;
		cameras[i].ppy = features[i].img_size.height / 2;
	}
	
	int points = 10;
	int skips = 0;
	vector<float> focals;
	for (int idx = 0; idx < pairwise_matches.size(); ++idx) {
		MatchesInfo &pw_matches = pairwise_matches[idx];
		int i = pw_matches.src_img_idx;
		int j = pw_matches.dst_img_idx;
		if (i >= j || pw_matches.confidence < CONF_THRESH) {
			++skips;
			continue;
		}
		int matches = pw_matches.matches.size();
		if (!matches) {
			++skips;
			continue;
		}
		for (int k = 0; k < points; ++k) {
			int match_idx = rand() % matches;
			int idx1 = pw_matches.matches[match_idx].queryIdx;
			int idx2 = pw_matches.matches[match_idx].trainIdx;
			KeyPoint k1 = features[i].keypoints[idx1];
			KeyPoint k2 = features[j].keypoints[idx2];
			float x1 = k1.pt.x - features[i].img_size.width / 2;
			float y1 = k1.pt.y - features[i].img_size.height / 2;
			float x2 = k2.pt.x - features[j].img_size.width / 2;
			float y2 = k2.pt.y - features[j].img_size.height / 2;
			float best_f;
			float best_err = std::numeric_limits<float>::max();
			float err;
			float f = 500;
			float delta = 10;
			float decay = 0.99;
			float err_thresh = 1;
			while (1) {
				f += delta;
				float x1_ = f * atan(x1 / f);
				float theta = j - i;
				if (i == 0 && j == NUM_IMAGES - 1 && wrapAround) {
					theta = -1;
				}
				theta *= 2 * PI / 6;
				float x2_ = f * (theta + atan(x2 / f));
				float y1_ = f * y1 / sqrt(x1*x1 + f*f);
				float y2_ = f * y2 / sqrt(x2*x2 + f*f);
				err = sqrt((x1_ - x2_)*(x1_ - x2_) + (y1_ - y2_)*(y1_ - y2_));
				if (err < best_err) {
					best_f = f;
					best_err = err;
				}
				else {
					f = best_f;
					delta *= -1;
				}
				if (err < err_thresh || abs(delta) < 0.1) {
					break;
				}
				delta *= decay;
			}
			float dist = pw_matches.matches[match_idx].distance;
			focals.push_back(best_f);
		}
	}
	
	std::sort(focals.begin(), focals.end());

	if (focals.size() % 2 == 1)
	{
		warped_image_scale = (focals[focals.size() / 2]);
	}
	else
	{
		warped_image_scale = (focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
	}

	for (int i = 0; i < cameras.size(); ++i) {
		cameras[i].focal = warped_image_scale;
	}
	return true;
}

// Precalculates warp maps for images so only precalculated maps are used to warp online
void warpImages(vector<Mat> full_img, Size full_img_size, vector<CameraParams> cameras, Ptr<Blender> blender, Ptr<ExposureCompensator> compensator, double work_scale,
				double seam_scale, double seam_work_aspect, vector<cuda::GpuMat> &x_maps, vector<cuda::GpuMat> &y_maps, double &compose_scale,
				float &warped_image_scale,  float &blend_width) {
	//times[0] = std::chrono::high_resolution_clock::now();
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

	//times[1] = std::chrono::high_resolution_clock::now();
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
	//times[1] = std::chrono::high_resolution_clock::now();

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
	else if (BLEND_TYPE == Blender::MULTI_BAND)
	{
		MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
		mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
	}
	else if (BLEND_TYPE == Blender::FEATHER)
	{
		FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
		fb->setSharpness(1.f / blend_width);
	}

	blender->prepare(corners, sizes);

	//times[2] = std::chrono::high_resolution_clock::now();

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

		// Blend the current image
		mb->init_gpu(gpu_img, gpu_mask_warped, corners[img_idx]);
	}
	//times[3] = std::chrono::high_resolution_clock::now();


	Mat result;
	Mat result_mask;

	blender->blend(result, result_mask);

	//times[4] = std::chrono::high_resolution_clock::now();
}


void calibrateMeshWarp(vector<Mat> &full_imgs, vector<cuda::GpuMat> &x_mesh, vector<cuda::GpuMat> &y_mesh,
					   vector<cuda::GpuMat> &x_maps, vector<cuda::GpuMat> &y_maps, double compose_scale) {
	int64 t = getTickCount();
	vector<Mat> images(full_imgs.size());
	Mat x_map;
	Mat y_map;
	for (int i = 0; i < full_imgs.size(); ++i) {
		LOGLN("Suus: " << (getTickCount() - t) / getTickFrequency() * 1000);
		t = getTickCount();
		resize(full_imgs[i], images[i], Size(), compose_scale, compose_scale);
		x_maps[i].download(x_map);
		y_maps[i].download(y_map);
		remap(images[i], images[i], x_map, y_map, INTER_LINEAR);
		Size mesh_size = images[i].size();
		cuda::GpuMat small_mesh_x;
		cuda::GpuMat small_mesh_y;
		x_mesh[i] = cuda::GpuMat(mesh_size, CV_32FC1);
		y_mesh[i] = cuda::GpuMat(mesh_size, CV_32FC1);
		LOGLN("Suus: " << (getTickCount() - t) / getTickFrequency() * 1000);
		t = getTickCount();
		int N = 10;
		int M = 10;
		Mat mesh_cpu_x(N, M, CV_32FC1);
		Mat mesh_cpu_y(N, M, CV_32FC1);
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < M; ++j) {
				mesh_cpu_x.at<float>(i, j) = j * mesh_size.width / N;
				mesh_cpu_y.at<float>(i, j) = i * mesh_size.height / M;
				if (i < 10 & i > 0) {
					mesh_cpu_y.at<float>(i, j) = abs(M / 2 - i) + i * mesh_size.height / M;
				}
			}
		}
		LOGLN("Suus: " << (getTickCount() - t) / getTickFrequency() * 1000);
		t = getTickCount();
		small_mesh_x.upload(mesh_cpu_x);
		small_mesh_y.upload(mesh_cpu_y);
		cuda::resize(small_mesh_x, x_mesh[i], mesh_size);
		cuda::resize(small_mesh_y, y_mesh[i], mesh_size);
		LOGLN("Suus: " << (getTickCount() - t) / getTickFrequency() * 1000);
	}
	LOGLN("Suus: " << (getTickCount() - t) / getTickFrequency() * 1000);
}


  // Takes in maps for 3D remapping, compose scale for sizing final panorama, blender and image size.
  // Returns true if all the phases of calibration are successful.
bool stitch_calib(vector<vector<Mat>> full_img, vector<CameraParams> &cameras, vector<cuda::GpuMat> &x_maps, vector<cuda::GpuMat> &y_maps, vector<cuda::GpuMat> &x_mesh,
	vector<cuda::GpuMat> &y_mesh, double &work_scale, double &seam_scale, double &seam_work_aspect, double &compose_scale, Ptr<Blender> &blender,
	Ptr<ExposureCompensator> compensator, float &warped_image_scale, float &blend_width, Size &full_img_size)
{
	// STEP 1: reading images, feature finding and matching // ------------------------------------------------------------------
	times[0] = std::chrono::high_resolution_clock::now();
	vector<ImageFeatures> features(NUM_IMAGES);

	for (int j = 0; j < full_img.size(); ++j) {
		for (int i = 0; i < NUM_IMAGES; ++i) {
			//CAPTURES[i].read(full_img[i]);
			if (full_img[j][i].empty()) {
				LOGLN("Can't read frame from camera/file nro " << i + j * NUM_IMAGES << "...");
				return false;
			}
		}
	}
	full_img_size = full_img[0][0].size();

	findFeatures(full_img, features, work_scale, seam_scale, seam_work_aspect);
	times[1] = std::chrono::high_resolution_clock::now();

	vector<MatchesInfo> pairwise_matches(NUM_IMAGES -1 + (int)wrapAround);

	matchFeatures(features, pairwise_matches);

	times[2] = std::chrono::high_resolution_clock::now();
	// STEP 2: estimating homographies // ---------------------------------------------------------------------------------------
	if (!calibrateCameras(features, pairwise_matches, cameras, warped_image_scale)) {
		return false;
	}

	times[3] = std::chrono::high_resolution_clock::now();

	warpImages(full_img[full_img.size()-1], full_img_size, cameras, blender, compensator, work_scale, seam_scale, seam_work_aspect,
			   x_maps, y_maps, compose_scale, warped_image_scale, blend_width);
	times[3] = std::chrono::high_resolution_clock::now();

	calibrateMeshWarp(full_img[full_img.size()-1], x_mesh, y_mesh, x_maps, y_maps, compose_scale);
	times[4] = std::chrono::high_resolution_clock::now();
	return true;
}

vector<cuda::GpuMat> full_imgs(NUM_IMAGES);
vector<cuda::GpuMat> images(NUM_IMAGES);
int printing = 0;
// Online stitching fuction, which resizes if necessary, remaps to 3D and uses the stripped version of feed function
void stitch_online(double compose_scale, Mat &img, cuda::GpuMat &x_map, cuda::GpuMat &y_map, cuda::GpuMat &x_mesh, cuda::GpuMat &y_mesh, 
							MultiBandBlender* mb, GainCompensator* gc, int thread_num)
{
	int img_num = thread_num % NUM_IMAGES;
	if (img_num == printing) {
		times[0] = std::chrono::high_resolution_clock::now();
	}
	cuda::Stream stream;


	// Upload to gpu memory
	full_imgs[img_num].upload(img, stream);

	if (img_num == printing) {
		times[1] = std::chrono::high_resolution_clock::now();
	}

	// Resize if necessary
	if (abs(compose_scale - 1) > 1e-1)
	{
		cuda::resize(full_imgs[img_num], images[img_num], Size(), compose_scale, compose_scale, 1, stream);

		if (img_num == printing) {
			times[2] = std::chrono::high_resolution_clock::now();
		}

		// Warp using existing maps
		cuda::remap(images[img_num], images[img_num], x_map, y_map, INTER_LINEAR, BORDER_REFLECT, Scalar(), stream);
	}
	else
	{
		// Warp using existing maps
		cuda::remap(full_imgs[img_num], images[img_num], x_map, y_map, INTER_LINEAR, BORDER_REFLECT, Scalar(), stream);
	}
	gc->apply_gpu(img_num, Point(), images[img_num], cuda::GpuMat());

	cuda::remap(images[img_num], images[img_num], x_mesh, y_mesh, INTER_LINEAR, BORDER_CONSTANT, Scalar(0), stream);
	
	if (img_num == printing) {
		times[3] = std::chrono::high_resolution_clock::now();
	}
	
	// Calculate pyramids for blending
	mb->feed_online(images[img_num], img_num, stream);

	if (img_num == printing) {
		times[4] = std::chrono::high_resolution_clock::now();
	}
}

void stitch_one(double compose_scale, vector<Mat> &imgs, vector<cuda::GpuMat> &x_maps, vector<cuda::GpuMat> &y_maps, vector<cuda::GpuMat> &x_mesh, vector<cuda::GpuMat> &y_mesh,
				MultiBandBlender* mb, GainCompensator* gc, BlockingQueue<cuda::GpuMat> &results) {
	for (int i = 0; i < NUM_IMAGES; ++i) {
		stitch_online(compose_scale, std::ref(imgs[i]), std::ref(x_maps[i]), std::ref(y_maps[i]), std::ref(x_mesh[i]), std::ref(y_mesh[i]), mb, gc, i);
	}
	LOGLN("Frame:::::");
	for (int i = 0; i < TIMES-1; ++i) {
		LOGLN("delta time: " << std::chrono::duration_cast<std::chrono::milliseconds>(times[i+1] - times[i]).count());
	}
	Mat result;
	Mat result_mask;
	cuda::GpuMat out;
	times[0] = std::chrono::high_resolution_clock::now();
	mb->blend(result, result_mask, out, true);
	times[1] = std::chrono::high_resolution_clock::now();
	LOGLN("delta time: " << std::chrono::duration_cast<std::chrono::milliseconds>(times[1] - times[0]).count());
	results.push(out);
}

void consume(BlockingQueue<cuda::GpuMat> &results) {
	cuda::GpuMat mat;
	int i = 0;
	VideoWriter outVideo;
	if (save_video) {
		outVideo.open("stitched.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(1920, 1080));
		if (!outVideo.isOpened()) {
			return;
		}
	}
	while (1) {
		Mat out;
		Mat small;
		mat = results.pop();
		if (mat.empty()) {
			if (save_video) {
				outVideo.release();
			}
			return;
		}
		mat.download(out);
		if (!i) {
			imwrite("calib.jpg", out);
		}
		resize(out, small, Size(1920, 1080));
		small.convertTo(small, CV_8U);
		//outVideo << small;
		imshow("Video", small);
		waitKey(1);
		++i;
	}
}

bool getImages(vector<VideoCapture> caps, vector<Mat> &images, int skip=0) {
	for (int i = 0; i < NUM_IMAGES; ++i) {
		Mat mat;
		bool capped;
		for (int j = 0; j < skip; ++j) {
			capped = caps[i].read(mat);
			if (!capped) {
				return false;
			}
		}
		capped = caps[i].read(mat);
		if (!capped) {
			return false;
		}
		images.push_back(mat);
	}
	return true;
}

int main(int argc, char* argv[])
{
	LOGLN("");
	//cuda::printCudaDeviceInfo(0);
	cuda::printShortCudaDeviceInfo(0);

	// Videofeed input --------------------------------------------------------
	
	for(int i = 0; i < NUM_IMAGES; ++i) {
		CAPTURES.push_back(VideoCapture(video_files[i]));
		if(!CAPTURES[i].isOpened()) {
			LOGLN("ERROR: Unable to open videofile(s).");
			return -1;
		}
		CAPTURES[i].set(CV_CAP_PROP_POS_FRAMES, skip_frames + offsets[i]);
	}
	// ------------------------------------------------------------------------

	// OFFLINE CALIBRATION // ---------------------------------------------------------------------------------------------------

	vector<cuda::GpuMat> x_maps(NUM_IMAGES);
	vector<cuda::GpuMat> y_maps(NUM_IMAGES);
	vector<cuda::GpuMat> x_mesh(NUM_IMAGES);
	vector<cuda::GpuMat> y_mesh(NUM_IMAGES);
	Size full_img_size;
	
	double work_scale = 1;
	double seam_scale = 1;
	double seam_work_aspect = 1;
	double compose_scale = 1;
	float blend_width = 0;
	float warped_image_scale;

	cuda::setDevice(0);

	Ptr<Blender> blender = Blender::createDefault(Blender::MULTI_BAND, true);
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	vector<Point> corners(NUM_IMAGES);
	vector<Size> sizes(NUM_IMAGES);
	vector<CameraParams> cameras;
	vector<vector<Mat>> full_img(INIT_FRAME_AMT);
	for (int i = 0; i < INIT_FRAME_AMT; ++i) {
		if (!getImages(CAPTURES, full_img[i], INIT_SKIPS)) {
			LOGLN("Couldn't read images");
			return -1;
		}
	}
	//if (!stitch_calib(x_maps, y_maps, compose_scale, blender, blend_width, full_img_size, corners, sizes))
	if (!stitch_calib(full_img, cameras, x_maps, y_maps, x_mesh, y_mesh, work_scale, seam_scale, seam_work_aspect, compose_scale, blender, compensator, warped_image_scale, blend_width, full_img_size))
	{
		LOGLN("");
		LOGLN("Calibration failed!");
		return -1;
	}
	blender = Blender::createDefault(Blender::MULTI_BAND, true);
	int64 start = getTickCount();
	if (!stitch_calib(full_img, cameras, x_maps, y_maps, x_mesh, y_mesh, work_scale, seam_scale, seam_work_aspect, compose_scale, blender, compensator, warped_image_scale, blend_width, full_img_size))
	{
		LOGLN("");
		LOGLN("Calibration failed!");
		return -1;
	}

	for (int i = 0; i < TIMES-1; ++i) {
		LOGLN("delta time: " << std::chrono::duration_cast<std::chrono::milliseconds>(times[i+1] - times[i]).count());
	}
	LOGLN("Calibration done in: " << (getTickCount() - start) / getTickFrequency() * 1000 << " ms");
	LOGLN("");
	LOGLN("Proceeding to online process...");
	LOGLN("");
	
	MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
	GainCompensator* gc = dynamic_cast<GainCompensator*>(compensator.get());

	// Temporary way to acquire images ----------------------------------------
	// Has to be adjusted manually
	/*
	// Doesn't exatly know if this is correct way to speed up uploading to gpu memory ---> 
	cuda::HostMem full_img_0_pl(full_img_size, CV_8UC3);
	cuda::HostMem full_img_1_pl(full_img_size, CV_8UC3);
	cuda::HostMem full_img_2_pl(full_img_size, CV_8UC3);
	cuda::HostMem full_img_3_pl(full_img_size, CV_8UC3);
	cuda::HostMem full_img_4_pl(full_img_size, CV_8UC3);
	cuda::HostMem full_img_5_pl(full_img_size, CV_8UC3);
	// <---
	Mat img_0 = full_img_0_pl.createMatHeader();
	Mat img_1 = full_img_1_pl.createMatHeader();
	Mat img_2 = full_img_2_pl.createMatHeader();
	Mat img_3 = full_img_3_pl.createMatHeader();
	Mat img_4 = full_img_4_pl.createMatHeader();
	Mat img_5 = full_img_5_pl.createMatHeader();
	img_0 = imread(FILES[0]);
	input.push_back(img_0);
	img_1 = imread(FILES[1]);
	input.push_back(img_1);
	img_2 = imread(FILES[2]);
	input.push_back(img_2);
	img_3 = imread(FILES[3]);
	input.push_back(img_3);
	img_4 = imread(FILES[4]);
	input.push_back(img_4);
	img_5 = imread(FILES[5]);
	input.push_back(img_5);
	*/
	//-------------------------------------------------------------------------
	BlockingQueue<cuda::GpuMat> results;
	int64 starttime = getTickCount();
	std::thread consumer(consume, std::ref(results));
	int frame_amt = 0;

	// ONLINE STITCHING // ------------------------------------------------------------------------------------------------------
	while(1) 
	{
		vector<Mat> input;
		bool capped = getImages(CAPTURES, input);
		if (!capped) {
			break;
		}
		if (frame_amt && (frame_amt % RECALIB_DEL == 0) && recalibrate) {
			int64 t = getTickCount();
			blender = Blender::createDefault(Blender::MULTI_BAND, true);
			//x_maps = vector<cuda::GpuMat>(NUM_IMAGES);
			//y_maps = vector<cuda::GpuMat>(NUM_IMAGES);
			//warpImages(input, full_img_size, cameras, blender, work_scale, seam_scale, seam_work_aspect, x_maps, y_maps,
			//		   compose_scale, warped_image_scale, blend_width);
			if (!stitch_calib(full_img, cameras, x_maps, y_maps, x_mesh, y_mesh, work_scale, seam_scale, seam_work_aspect, compose_scale, blender, compensator, warped_image_scale, blend_width, full_img_size))
			{
				LOGLN("");
				LOGLN("Calibration failed!");
				return -1;
			}
			mb = dynamic_cast<MultiBandBlender*>(blender.get());
			LOGLN("Rewarp: " << (getTickCount()-t)*1000/getTickFrequency());
		}

		stitch_one(compose_scale, input, x_maps, y_maps, x_mesh, y_mesh, mb, gc, results);
		++frame_amt;
	}

	int64 end = getTickCount();
	double delta = (end - starttime) / getTickFrequency() * 1000;
	std::cout << "Time taken: " << delta / 1000 << " seconds. Avg fps: " << frame_amt / delta * 1000 << std::endl;
	cuda::GpuMat matti;
	bool sis = matti.empty();
	results.push(matti);
	consumer.join();
	return 0;
}