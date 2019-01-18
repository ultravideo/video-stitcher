#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#define CAPTURE_TCP_PORT "6666"

/* The capture stream is in 1920*1080 resolution, YUV 4:2:0 NV12 format. The format has a 1 channel, 1920*1080 Y plane, and a 1 channel, 1920*540 (i.e. same width, half height) UV plane.
 * The UV plane is directly after the Y plane, so the data of one full image is 1 channel, 1920*1620 bytes. The image is converted to 3-channel, 1920*1080
 * BGR format after receiving it.
 */

#define CAPTURE_IMG_WIDTH 1920
#define CAPTURE_IMG_HEIGHT 1620
#define CAPTURE_IMG_CHANNELS 1

#define PLAYER_ADDRESS "localhost"
#define PLAYER_TCP_PORT "55555"

const std::string base = "videos/";
const std::string folder = base + "new";
const int skip_frames = 135;
const bool wrapAround = true;
const bool recalibrate = true;
const bool enable_local = true;
const bool save_video = false;
const bool show_out = true;
const bool use_stream = false;
const bool clear_buffers = false;
const bool debug_stream = false;
const bool use_surf = true;
const bool send_results = false;
const bool send_height_info = true;
const bool keep_aspect_ratio = true;
const bool add_black_bars = false; /* NOTE: this should be true if send_results is set to true */
const int NUM_IMAGES = 6;
const unsigned char clientAddrStart = 41; //The capture boards should have consecutive IP addresses, with the last octet starting from this number
const int OUTPUT_WIDTH = 4096;
const int OUTPUT_HEIGHT = 2048;
const unsigned int RESULTS_MAX_SIZE = 0;
//const int offsets[NUM_IMAGES] = {0, 37, 72, 72, 37}; // static
//const int offsets[NUM_IMAGES] = {43, 153, 151, 131, 95, 0}; // new
const int offsets[NUM_IMAGES] = { -92+134, 30+134, 28+134, 0+134, -40+134, -134+134 }; // test videos
//const int offsets[NUM_IMAGES] = { 0 }; // dynamic
const int RECALIB_DEL = 10;
const double WORK_MEGAPIX = 0.6;	//0.6;	//-1			// Megapix parameter is scaled to the number
const double SEAM_MEAGPIX = 0.01;							// of pixels in full image and this is used
const double COMPOSE_MEGAPIX = 1.4;	//1.4;	//2.2;	//-1	// as a scaling factor when resizing images
const float MATCH_CONF = 0.5f;
const float BLEND_STRENGTH = 5;
const int HESS_THRESH = 300;
const int NOCTAVES = 4;
const int NOCTAVESLAYERS = 2;

const int MAX_FEATURES_PER_IMAGE = 100;
const bool VISUALIZE_MATCHES = false; // Draw the meshes and matches to images pre mesh warp
const bool VISUALIZE_WARPED = false; // Draw the warped mesh
const int MESH_HEIGHT = 10;
const int MESH_WIDTH = 10;
// Alphas are weights for different cost functions
// 0: Local alignment, 1: Global alignment, 2: Smoothness
const float ALPHAS[3] = {1.0f, 0.01f, 0.00005f};
const int GLOBAL_DIST = 20; // Maximum distance from vertex in global warping

// Test material before right videos are obtained from the camera rig
const std::vector<std::string> video_files = {folder + "/0.mp4", folder + "/1.mp4", folder + "/2.mp4", folder + "/3.mp4", folder + "/4.mp4", folder + "/5.mp4"};

#define PI 3.1415926535897932384626
#define LOGLN(msg) std::cout << msg << std::endl

// Comment this out when on windows
//#define LINUX
