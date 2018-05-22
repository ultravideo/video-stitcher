#include "networking.h" // networking.h has to be included before Windows.h
#include <fstream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <vector>
#include <limits>
#include <iostream>
#include <string>
#include <Windows.h>

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

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>

#include "blockingqueue.h"
#include "calibration.h"


const int TIMES = 5;
std::chrono::high_resolution_clock::time_point times[TIMES];

using std::vector;

using namespace cv;
using namespace cv::detail;

vector<cuda::GpuMat> full_imgs(NUM_IMAGES);
vector<cuda::GpuMat> images(NUM_IMAGES);
vector<cuda::GpuMat> warped_images(NUM_IMAGES);
Ptr<cuda::Filter> filter = cuda::createGaussianFilter(CV_8UC3, CV_8UC3, Size(5, 5), 15);
int printing = 0;
// Online stitching fuction, which resizes if necessary, remaps to 3D and uses the stripped version of feed function
void stitch_online(double compose_scale, Mat &img, cuda::GpuMat &x_map, cuda::GpuMat &y_map,
                   cuda::GpuMat &x_mesh, cuda::GpuMat &y_mesh, MultiBandBlender* mb,
                   GainCompensator* gc, int thread_num)
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
		cuda::remap(images[img_num], images[img_num], x_map, y_map, INTER_LINEAR,
                    BORDER_CONSTANT, Scalar(0), stream);
	}
	else
	{
		// Warp using existing maps
		cuda::remap(full_imgs[img_num], images[img_num], x_map, y_map, INTER_LINEAR,
                    BORDER_CONSTANT, Scalar(0), stream);
	}
    // Apply gain compensation
	images[img_num].convertTo(images[img_num], images[img_num].type(), gc->gains()[img_num]);

    if (enable_local) {
        // Warp the image and the mask according to a mesh
        cuda::remap(images[img_num], warped_images[img_num], x_mesh, y_mesh,
                    INTER_LINEAR, BORDER_CONSTANT, Scalar(0), stream);
    }
    else
    {
        warped_images[img_num] = images[img_num];
    }
	
	//filter->apply(images[img_num], images[img_num], stream);
	if (img_num == printing) {
		times[3] = std::chrono::high_resolution_clock::now();
	}
	
	// Calculate pyramids for blending
	mb->feed_online(warped_images[img_num], img_num, stream);

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
		Mat small_img;
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
		resize(out, small_img, Size(1920, 1080));
		small_img.convertTo(small_img, CV_8U);
		//outVideo << small_img;
		imshow("Video", small_img);
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

bool getImages(vector<BlockingQueue<Mat>> &queues, vector<Mat> &images) {
	for (int i = 0; i < NUM_IMAGES; ++i) {
		images.push_back(queues[i].pop());
	}
	return true;
}

int main(int argc, char* argv[])
{
	vector<BlockingQueue<Mat>> que(NUM_IMAGES);
    std::vector<VideoCapture> CAPTURES;
    if (use_stream) {
        if (startPolling(que)) {
            return -1;
        }
    }
    if (debug_stream) {
        while (1) {
            for (int i = 0; i < NUM_IMAGES; ++i) {
                Mat mat = que[i].pop();
                imshow(std::to_string(i), mat);
            }
            waitKey(1);
        }
    }

	LOGLN("");
	//cuda::printCudaDeviceInfo(0);
	cuda::printShortCudaDeviceInfo(0);
	cuda::setDevice(0);

	// Videofeed input
    if (!use_stream) {
        for (int i = 0; i < NUM_IMAGES; ++i) {
            CAPTURES.push_back(VideoCapture(video_files[i]));
            if (!CAPTURES[i].isOpened()) {
                LOGLN("ERROR: Unable to open videofile(s).");
                return -1;
            }
            CAPTURES[i].set(CV_CAP_PROP_POS_FRAMES, skip_frames + offsets[i]);
        }
    }

	// OFFLINE CALIBRATION 
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

	Ptr<Blender> blender = Blender::createDefault(Blender::MULTI_BAND, true);
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	vector<Point> corners(NUM_IMAGES);
	vector<Size> sizes(NUM_IMAGES);
	vector<CameraParams> cameras;
	vector<Mat> full_img;

    bool ret;
    if (use_stream) {
        ret = getImages(que, full_img);
    } else {
        ret = getImages(CAPTURES, full_img);
    }
    if (!ret) {
        LOGLN("Couldn't read images");
        return -1;
    }

	int64 start = getTickCount();
	if (!stitch_calib(full_img, cameras, x_maps, y_maps, x_mesh, y_mesh, work_scale, seam_scale,
                      seam_work_aspect, compose_scale, blender, compensator, warped_image_scale,
                      blend_width, full_img_size))
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
	GainCompensator* gc = dynamic_cast<GainCompensator*>(compensator.get());

	//-------------------------------------------------------------------------
	BlockingQueue<cuda::GpuMat> results;
	int64 starttime = getTickCount();
	std::thread consumer(consume, std::ref(results));
	int frame_amt = 0;

	// ONLINE STITCHING // ------------------------------------------------------------------------------------------------------
	while(1) 
	{
		vector<Mat> input;
		bool capped;
		if (use_stream) {
			capped = getImages(que, input);
		} else {
			capped = getImages(CAPTURES, input);
		}
		if (!capped) {
			break;
		}

		if (frame_amt && (frame_amt % RECALIB_DEL == 0) && recalibrate) {
			int64 t = getTickCount();
            recalibrateMesh(input, x_maps, y_maps, x_mesh, y_mesh, cameras[0].focal, compose_scale);
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