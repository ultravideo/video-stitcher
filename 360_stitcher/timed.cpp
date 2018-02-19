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

#include <fstream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <vector>

#define LOGLN(msg) std::cout << msg << std::endl

using std::vector;

using namespace cv;
using namespace cv::detail;

std::mutex cout_mutex;

int const NUM_IMAGES = 3;			//2		//6
double const WORK_MEGAPIX = 0.6;	//0.6;	//-1			// Megapix parameter is scaled to the number
double const SEAM_MEAGPIX = 0.01;							// of pixels in full image and this is used
double const COMPOSE_MEGAPIX = 1.4;	//1.4;	//2.2;	//-1	// as a scaling factor when resizing images
float const MATCH_CONF = 0.3f;
float const CONF_THRESH = 1.0f;
int const BLEND_TYPE = Blender::MULTI_BAND;					// Feather blending leaves ugly seams
float const BLEND_STRENGTH = 5;
int const HESS_THRESH = 300;
int const NOCTAVES = 4;
int const NOCTAVESLAYERS = 2;
int skip_frames = 5;

// Test material before right videos are obtained from the camera rig
vector<VideoCapture> CAPTURES;

// Print function for parallel threads - debugging
void msg(String const message, double const value, int const thread_id)
{
	cout_mutex.lock();
	std::cout << thread_id << ": " << message << value << std::endl;
	cout_mutex.unlock();
}
void findFeatures(vector<Mat> &full_img, vector<Mat> &images, vector<ImageFeatures> &features,
				  double &work_scale, double &seam_scale, double &seam_work_aspect) {
	Ptr<FeaturesFinder> finder = makePtr<SurfFeaturesFinderGpu>();
	// Read images from file and resize if necessary
	for (int i = 0; i < NUM_IMAGES; i++)
	{
		// Negative value means processing images in the original size
		if (WORK_MEGAPIX < 0)
		{
			images[i] = full_img[i];
			work_scale = 1;
		}
		// Else downscale images to speed up the process
		else
		{
			work_scale = min(1.0, sqrt(WORK_MEGAPIX * 1e6 / full_img[i].size().area()));
			cv::resize(full_img[i], images[i], Size(), work_scale, work_scale);
		}

		// Calculate scale for downscaling done in seam finding process
		seam_scale = min(1.0, sqrt(SEAM_MEAGPIX * 1e6 / full_img[i].size().area()));
		seam_work_aspect = seam_scale / work_scale;

		// Find features with SURF feature finder
		(*finder)(images[i], features[i]);
		features[i].img_idx = i;

		// Resize images for seam process
		cv::resize(full_img[i], images[i], Size(), seam_scale, seam_scale);
	}
	finder->collectGarbage();
}
  // Takes in maps for 3D remapping, compose scale for sizing final panorama, blender and image size.
  // Returns true if all the phases of calibration are successful.
	bool stitch_calib(vector<cuda::GpuMat> &x_maps, vector<cuda::GpuMat> &y_maps, double &compose_scale, Ptr<Blender> &blender,
		float &blend_width, Size &full_img_size)
{
	// STEP 1: reading images, feature finding and matching // ------------------------------------------------------------------

	double work_scale = 1;
	double seam_scale = 1;
	double seam_work_aspect = 1;


	vector<Mat> full_img(NUM_IMAGES);
	vector<Mat> img(NUM_IMAGES);
	vector<Mat> images(NUM_IMAGES);
	vector<ImageFeatures> features(NUM_IMAGES);

	for (int i = 0; i < NUM_IMAGES; ++i) {
		CAPTURES[i].read(full_img[i]);
		if (full_img[i].empty()) {
			LOGLN("Can't read frame from camera/file nro " << i << "...");
			return false;
		}
	}
	full_img_size = full_img[0].size();

	findFeatures(full_img, images, features, work_scale, seam_scale, seam_work_aspect);

	vector<MatchesInfo> pairwise_matches;
	Ptr<FeaturesMatcher> matcher = makePtr<BestOf2NearestMatcher>(true, MATCH_CONF);

	// Match features
	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();

	/*
	// saving matches graph -file, which contains info from matching
	LOGLN("Saving matches graph...");
	std::ofstream f("graph.txt");
	vector<String> names = {"pic1", "pic2", "pic3" , "pic4" };
	f << matchesGraphAsString(names, pairwise_matches, CONF_THRESH);
	*/

	// STEP 2: estimating homographies // ---------------------------------------------------------------------------------------

	Ptr<Estimator> estimator = makePtr<HomographyBasedEstimator>();
	vector<CameraParams> cameras;

	// Rough estimation of the homographies
	if (!(*estimator)(features, pairwise_matches, cameras))
	{
		LOGLN("Homography estimation failed...");
		return false;
	}

	// Convert rotation matrix into a type: 32-bit float
#pragma omp parallel for
	for (int i = 0; i < cameras.size(); i++)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}

	Ptr<BundleAdjusterBase> adjuster = makePtr<BundleAdjusterRay>();
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	refine_mask(0, 0) = 1;
	refine_mask(0, 1) = 1;
	refine_mask(0, 2) = 1;
	refine_mask(1, 1) = 1;
	refine_mask(1, 2) = 1;

	adjuster->setConfThresh(CONF_THRESH);
	adjuster->setRefinementMask(refine_mask);

	// Calculate final estimation of the homographies
	if (!(*adjuster)(features, pairwise_matches, cameras))
	{
		LOGLN("Camera parameters adjusting failed...");
		return false;
	}

	// Calculation of the mean focal for the warping scale
	vector<double> focals((int)cameras.size());

#pragma omp parallel for
	for (int i = 0; i < cameras.size(); i++)
	{
		focals[i] = cameras[i].focal;
		std::cout << "R = " << std::endl << " " << cameras[i].R << std::endl << std::endl;
		std::cout << "t = " << std::endl << " " << cameras[i].t << std::endl << std::endl;
	}

	sort(focals.begin(), focals.end());

	float warped_image_scale;

	if (focals.size() % 2 == 1)
	{
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	}
	else
	{
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
	}

	// Do wave correct i.e. try to make panorama more lined-up horizontally
	vector<Mat> rmats((int)cameras.size());
#pragma omp parallel for
	for (int i = 0; i < cameras.size(); i++)
	{
		rmats[i] = cameras[i].R.clone();
	}

	waveCorrect(rmats, WAVE_CORRECT_HORIZ);

#pragma omp parallel for
	for (int i = 0; i < cameras.size(); i++)
	{
		cameras[i].R = rmats[i];
	}

	// STEP 3: warping images // ------------------------------------------------------------------------------------------------

	vector<UMat> masks_warped(NUM_IMAGES);
	vector<UMat> images_warped(NUM_IMAGES);
	vector<UMat> masks(NUM_IMAGES);
	vector<Size> sizes(NUM_IMAGES);

	// Create masks for warping
#pragma omp parallel for
	for (int i = 0; i < NUM_IMAGES; i++)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarperGpu>();
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

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

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);

		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	}

	// STEP 4: compensating exposure and finding seams // -----------------------------------------------------------------------

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	compensator->feed(corners, images_warped, masks_warped);

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
	warped_image_scale *= static_cast<float>(compose_work_aspect);
	warper = warper_creator->create(warped_image_scale);

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

	Mat img_warped;
	Mat img_warped_s;
	Mat dilated_mask;
	Mat seam_mask;
	Mat mask;

	vector<cuda::GpuMat> gpu_mask_warped(NUM_IMAGES);
	Size img_size;

	for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
	{
		if (abs(compose_scale - 1) > 1e-1)
		{
			resize(full_img[img_idx], img[img_idx], Size(), compose_scale, compose_scale);
		}
		else
		{
			img[img_idx] = full_img[img_idx];
		}

		full_img[img_idx].release();

		img_size = img[img_idx].size();

		Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);

		// Warp the current image
		warper->warp(img[img_idx], K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		Mat x_map;
		Mat y_map;

		// Create warping map for online process
		warper->buildMaps(img[img_idx].size(), K, cameras[img_idx].R, x_map, y_map);

		// Upload to gpu memory
		x_maps[img_idx].upload(x_map);
		y_maps[img_idx].upload(y_map);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));

		vector<Mat> mask_warped(NUM_IMAGES);

		warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped[img_idx]);

		// Apply exposure compensation
		compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped[img_idx]);

		img_warped.convertTo(img_warped_s, CV_16S);

		img_warped.release();
		img[img_idx].release();
		mask.release();

		cv::dilate(masks_warped[img_idx], dilated_mask, Mat());
		cv::resize(dilated_mask, seam_mask, mask_warped[img_idx].size());
		mask_warped[img_idx] = seam_mask & mask_warped[img_idx];
		gpu_mask_warped[img_idx].upload(masks_warped[img_idx]);

		// Blend the current image
		blender->feed(img_warped_s, mask_warped[img_idx], corners[img_idx]);
	}

	Mat result;
	Mat result_mask;

	blender->blend(result, result_mask);

	// WHAT HAPPEN TO THE FINAL PANORAMA? - writing to file is only here for debugging 
	cv::imwrite("calib.jpg", result);

	return true;
}
vector<cuda::GpuMat> full_imgs(NUM_IMAGES);
vector<cuda::GpuMat> images(NUM_IMAGES);
int printing = -1;
// Online stitching fuction, which resizes if necessary, remaps to 3D and uses the stripped version of feed function
void stitch_online(double compose_scale, Mat &img, cuda::GpuMat &x_map, cuda::GpuMat &y_map, 
							MultiBandBlender* mb, int thread_num)
{
	int img_num = thread_num % NUM_IMAGES;
	cuda::Stream stream;

//	int64 t;
//	int64 start = getTickCount();

	// Upload to gpu memory
	full_imgs[img_num].upload(img, stream);

//	if (img_num == printing) {
//		msg("uploading took (ms): ", (getTickCount() - start) / getTickFrequency() * 1000, img_num);
//		t = getTickCount();
//	}

	// Resize if necessary
	if (abs(compose_scale - 1) > 1e-1)
	{
		cuda::resize(full_imgs[img_num], images[img_num], Size(), compose_scale, compose_scale, 1, stream);

//		if (img_num == printing) {
//			msg("resizing took (ms): ", (getTickCount() - t) / getTickFrequency() * 1000, img_num);
//			t = getTickCount();
//		}

		// Warp using existing maps
		cuda::remap(images[img_num], images[img_num], x_map, y_map, INTER_LINEAR, BORDER_REFLECT, Scalar(), stream);
	}
	else
	{
		// Warp using existing maps
		cuda::remap(full_imgs[img_num], images[img_num], x_map, y_map, INTER_LINEAR, BORDER_REFLECT, Scalar(), stream);
	}
	
//	if (img_num == printing) {
//		msg("remapping took (ms): ", (getTickCount() - t) / getTickFrequency() * 1000, img_num);
//		t = getTickCount();
//	}
	
	// Calculate pyramids for blending
	mb->feed_online(images[img_num], img_num, stream);

//	if(img_num == printing)
//		msg("pyramid calculation took (ms): ", (getTickCount() - t) / getTickFrequency() * 1000, img_num);
	
//	if(img_num == printing)
//		msg("thread run (ms): ", (getTickCount() - start) / getTickFrequency() * 1000, img_num);
}

void stitch_one(double compose_scale, vector<Mat> &imgs, vector<cuda::GpuMat> &x_maps, vector<cuda::GpuMat> &y_maps,
				MultiBandBlender* mb, BlockingQueue<cuda::GpuMat> &results) {
	for (int i = 0; i < NUM_IMAGES; ++i) {
		stitch_online(compose_scale, std::ref(imgs[i]), std::ref(x_maps[i]), std::ref(y_maps[i]), mb, i);
	}
	Mat result;
	Mat result_mask;
	cuda::GpuMat out;
	int64 t = getTickCount();
	mb->blend(result, result_mask, out, true);
//	LOGLN("blending: " << (getTickCount()-t)/getTickFrequency()*1000);
	results.push(out);
}

void consume(BlockingQueue<cuda::GpuMat> &results) {
	cuda::GpuMat mat;
	while (1) {
		Mat out;
		Mat small;
		mat = results.pop();
		mat.download(out);
		out.convertTo(small, CV_8U);
		imshow("Video", small);
		waitKey(1);
	}
}

bool getImages(vector<VideoCapture> caps, vector<Mat> &images) {
	for (int i = 0; i < NUM_IMAGES; ++i) {
		Mat mat;
		bool capped = caps[i].read(mat);
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
	
	VideoCapture capture0("videos/3.mp4");
	VideoCapture capture1("videos/4.mp4");
	VideoCapture capture2("videos/5.mp4");

	if ((!capture0.isOpened()) || (!capture1.isOpened()))
	{
		LOGLN("ERROR: Unable to open videofile(s).");
		return -1;
	}

	capture0.set(CV_CAP_PROP_POS_FRAMES, skip_frames);
	capture1.set(CV_CAP_PROP_POS_FRAMES, skip_frames);
	capture2.set(CV_CAP_PROP_POS_FRAMES, skip_frames);

	CAPTURES.push_back(capture0);
	CAPTURES.push_back(capture1);
	CAPTURES.push_back(capture2);
	// ------------------------------------------------------------------------

	// OFFLINE CALIBRATION // ---------------------------------------------------------------------------------------------------
	int64 start = getTickCount();

	vector<cuda::GpuMat> x_maps(NUM_IMAGES);
	vector<cuda::GpuMat> y_maps(NUM_IMAGES);
	Size full_img_size;
	
	double compose_scale = 1;
	float blend_width = 0;

	cuda::setDevice(0);

	Ptr<Blender> blender = Blender::createDefault(Blender::MULTI_BAND, true);
	vector<Point> corners(NUM_IMAGES);
	vector<Size> sizes(NUM_IMAGES);

	//if (!stitch_calib(x_maps, y_maps, compose_scale, blender, blend_width, full_img_size, corners, sizes))
	if (!stitch_calib(x_maps, y_maps, compose_scale, blender, blend_width, full_img_size))
	{
		LOGLN("");
		LOGLN("Calibration failed!");
		return -1;
	}

	LOGLN("Calibration done in: " << (getTickCount() - start) / getTickFrequency() * 1000 << " ms");
	LOGLN("");
	LOGLN("Proceeding to online process...");
	LOGLN("");

	MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());

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
	consumer.detach();
	int frame_amt = 0;

	// ONLINE STITCHING // ------------------------------------------------------------------------------------------------------
	while(1) 
	{
		vector<Mat> input;
		bool capped = getImages(CAPTURES, input);
		if (!capped) {
			break;
		}

		stitch_one(compose_scale, input, x_maps, y_maps, mb, results);
		++frame_amt;
	}

	int64 end = getTickCount();
	double delta = (end - starttime) / getTickFrequency() * 1000;
	std::cout << "Time taken: " << delta / 1000 << " seconds. Avg fps: " << frame_amt / delta * 1000 << std::endl;
	return 0;
}