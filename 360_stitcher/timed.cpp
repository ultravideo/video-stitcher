#include <fstream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <vector>
#include <limits>
#include <iostream>
#include <string>
#include <kvazaar.h>

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

#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"

#include "networking.h" 
#include "netlib.h"
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

	//ilter->apply(images[img_num], images[img_num], stream);
	if (img_num == printing) {
		times[3] = std::chrono::high_resolution_clock::now();
	}

	// Calculate pyramids for blending
	mb->feed_online(warped_images[img_num], img_num, stream);

	if (img_num == printing) {
		times[4] = std::chrono::high_resolution_clock::now();
	}
}

void stitch_one(double compose_scale, vector<Mat> &imgs, vector<cuda::GpuMat> &x_maps,
               vector<cuda::GpuMat> &y_maps, vector<cuda::GpuMat> &x_mesh, vector<cuda::GpuMat> &y_mesh,
	           MultiBandBlender* mb, GainCompensator* gc, BlockingQueue<cuda::GpuMat> &results)
{
	for (int i = 0; i < NUM_IMAGES; ++i) {
		stitch_online(compose_scale, std::ref(imgs[i]), std::ref(x_maps[i]), std::ref(y_maps[i]),
                      std::ref(x_mesh[i]), std::ref(y_mesh[i]), mb, gc, i);
	}

	Mat result;
	Mat result_mask;
	cuda::GpuMat out;
	times[0] = std::chrono::high_resolution_clock::now();
	mb->blend(result, result_mask, out, true);
	times[1] = std::chrono::high_resolution_clock::now();

	if (clear_buffers) {
		while (!results.empty()) {
			results.pop();
		}
	}

	// Don't push to results if there are enough frames already. This is to prevent memory overflow,
    // if the frames are not consumed fast enough. Max size 0 means no limit.
	if (RESULTS_MAX_SIZE == 0 || results.size() < RESULTS_MAX_SIZE) {
		results.push(out);
	}
}

/* pc player: stitcher is client and player is server
 * android player: stitcher is server and player is client */
void connect_to_player(sts_net_socket_t *server, sts_net_socket_t *client)
{
    sts_net_close_socket(server);
    sts_net_close_socket(client);

#ifdef PC_PLAYER
		if (sts_net_open_socket(client, PLAYER_ADDRESS, PLAYER_TCP_PORT) < 0) {
            fprintf(stderr, "failed to open client socket: %s\n", sts_net_get_last_error());
            exit(EXIT_FAILURE);
		}
#else
        LOGLN("waiting for android player to connect...");

        if (sts_net_open_socket(server, NULL, PLAYER_TCP_PORT) < 0) {
            fprintf(stderr, "failed to open server socket: %s\n", sts_net_get_last_error());
            exit(EXIT_FAILURE);
        }

        
        if (sts_net_accept_socket(server, client) < 0) {
            fprintf(stderr, "failed to open client socket: %s\n", sts_net_get_last_error());
            exit(EXIT_FAILURE);
        }
#endif
}

void consume(BlockingQueue<cuda::GpuMat> &results)
{
	cuda::GpuMat mat;
	cuda::GpuMat mat_8u;
	bool first_frame = true;
	VideoWriter outVideo;
	Mat original_8u;
	Mat resized_bgr;
	Mat resized_rgb;

	//Initialize final_result as a black image
	Mat final_result = Mat(Size(OUTPUT_WIDTH, OUTPUT_HEIGHT), CV_8UC3, cv::Scalar(0));
	if (save_video) {
		outVideo.open("stitched.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(1920, 1080));
		if (!outVideo.isOpened()) {
            return;
		}
	}
    sts_net_socket_t server, socket;
	int frame_counter = 0;
	int iSendResult;

    kvz_config *config = NULL;
    kvz_encoder *enc   = NULL;
    kvz_picture *pic   = NULL;
    kvz_api *api       = NULL;

	if (send_results) {
        sts_net_init();
        connect_to_player(&server, &socket);

        api = const_cast<kvz_api *>(kvz_api_get(8));

        if ((config = api->config_alloc()) == NULL) {
            LOGLN("ERROR: failed to allocate config");
            exit(EXIT_FAILURE);
        }
        api->config_init(config);

        config->width = OUTPUT_WIDTH;
        config->height = OUTPUT_HEIGHT;
        api->config_parse(config, "preset", "ultrafast");

        if ((enc = api->encoder_open(config)) == NULL) {
            LOGLN("ERROR: failed to open encoder");
            exit(EXIT_FAILURE);
        }

        if ((pic = api->picture_alloc(config->width, config->height)) == NULL) {
            LOGLN("ERROR: failed to allocate picture");
            exit(EXIT_FAILURE);
        }
        pic->pts = 0;
	}

	Size resize_dst_size;
	uchar *row_ptr = final_result.ptr(0);
	std::chrono::high_resolution_clock::time_point tp = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::time_point tp2;
    int64_t num_frames, max;

	int sent_bytes = 0;
	int total_bytes = 0;

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	while (1) {
		mat = results.pop();
		if (mat.empty()) {
			if (save_video) {
				outVideo.release();
			}
            break;
		}

		mat.convertTo(mat_8u, CV_8U);
		mat_8u.download(original_8u);

		if (first_frame) {
			imwrite("calib.jpg", original_8u);
			int image_height;
			if (keep_aspect_ratio) {
				//Assume that we are resizing a very wide image, so that output width is the more restricting dimension.
				//Resize the width to exactly the correct size, and use the ratio between output and input widths to resize the height.
				//Cols = width (number of columns) and rows = height (number of rows)
				image_height = (double)OUTPUT_WIDTH / (double)original_8u.cols * original_8u.rows + 0.5;
				if (image_height > OUTPUT_HEIGHT) {
					image_height = OUTPUT_HEIGHT;
				}

                fprintf(stderr, "image height %d\n", image_height);
			}
			else {
				image_height = OUTPUT_HEIGHT;
			}
			resize_dst_size = Size(OUTPUT_WIDTH, image_height);

		}

		resize(original_8u, resized_bgr, resize_dst_size, 0, 0, INTER_LINEAR);
		if (keep_aspect_ratio && add_black_bars) {
			cvtColor(resized_bgr, resized_rgb, COLOR_BGR2RGB);

			// To add black bars, the stitched image is copied to the middle of a black image
			if (first_frame) {
				row_ptr = final_result.ptr(final_result.rows / 2 - resized_rgb.rows / 2);
			}
			memcpy(row_ptr, resized_rgb.data, resized_rgb.u->size);
		} else {
			cvtColor(resized_bgr, final_result, COLOR_BGR2RGB);
		}

		if (first_frame) {
			total_bytes = final_result.u->size;
			if (send_results && send_height_info) {
				//Tell the image height to the player. This has to be done, because the height can vary between different runs.
                //The player needs this information to place the image correctly on the sphere.
				int result_height = final_result.rows;

                if (sts_net_send(&socket, (char *)&result_height, sizeof(int)) <= 0) {
                    fprintf(stderr, "sending image height failed with error: %s\n", sts_net_get_last_error());
                    exit(EXIT_FAILURE);
                }
			}
		}

		if (send_results) {
            Mat final_result_yuv;
            cvtColor(final_result, final_result, COLOR_BGR2RGB);
            cvtColor(final_result, final_result_yuv, COLOR_BGR2YUV_I420);

            if (add_black_bars) {
                memcpy(pic->fulldata, final_result_yuv.data,
                        (final_result.rows * final_result.cols) + (final_result.rows * final_result.cols / 2));
            } else {
                /* TODO:  */
            }

            uint64_t nwritten  = 0;
            int32_t bytes_sent = 0;
            kvz_data_chunk *chunks_out = NULL;

            pic->pts++;

            /* call encoder_encode until we get chunks, set img_in to NULL after first call */
            for (kvz_picture *img_in = pic; chunks_out == NULL; img_in = NULL) {
                api->encoder_encode(enc, img_in, &chunks_out, NULL, NULL, NULL, NULL);
            }

            for (kvz_data_chunk *chunk = chunks_out; chunk != NULL; chunk = chunk->next) {
                int num_bytes = sts_net_send(&socket, chunk->data, chunk->len);

                if (num_bytes <= 0) {
                    fprintf(stderr, "sending data failed with error: %s\n", sts_net_get_last_error());
                    fprintf(stderr, "trying to reinitialize connection...\n");

                    /* for android player this will block until the connection is re-established */
                    connect_to_player(&server, &socket);

                    /* reopen encoder to get valid bitstream again
                     * (TODO: check with Marko if this is OK) */
                    api->encoder_close(enc);
                    if ((enc = api->encoder_open(config)) == NULL) {
                        LOGLN("ERROR: failed to open encoder");
                        exit(EXIT_FAILURE);
                    }
                }
            }
            api->chunk_free(chunks_out);
            LOGLN("encoded frame");
		}

		if (save_video) {
			outVideo << final_result;
		}

		if (first_frame) {
			imwrite("calib_resized.jpg", resized_bgr);
			imwrite("result.jpg", final_result);
			first_frame = false;
		}

		if (show_out) {
			cvtColor(final_result, final_result, CV_BGR2RGB);
			imshow("Video", final_result);
			waitKey(1);
		}

		++frame_counter;
		if (frame_counter == 30) {
			tp2 = std::chrono::high_resolution_clock::now();
			LOGLN("delta time 30 frames: " << std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp).count() << " ms");
			frame_counter = 0;
			tp = std::chrono::high_resolution_clock::now();
		}
	}
}

bool getImages(vector<VideoCapture> caps, vector<Mat> &images, int skip = 0) {
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
	if (clear_buffers) {
		images.clear();
	}
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
				if (!que[i].empty()) {
					if (show_out) {
						imshow(std::to_string(i), que[i].pop());
						waitKey(1);
					}
					else {
						que[i].pop();
					}
				}
			}
			//waitKey(1);
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
	}
	else {
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
	int frame_amt = 0;
    
 	std::thread consumer(consume, std::ref(results));

	// ONLINE STITCHING // ------------------------------------------------------------------------------------------------------
	while (1)
	{
		vector<Mat> input;
		bool capped;
		if (use_stream) {
			capped = getImages(que, input);
		}
		else {
			capped = getImages(CAPTURES, input);
		}
		if (!capped) {
			break;
		}

		if (frame_amt && (frame_amt % RECALIB_DEL == 0) && recalibrate) {
			int64 t = getTickCount();
			recalibrateMesh(input, x_maps, y_maps, x_mesh, y_mesh, cameras[0].focal, compose_scale, work_scale);
			LOGLN("Rewarp: " << (getTickCount() - t) * 1000 / getTickFrequency());
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
