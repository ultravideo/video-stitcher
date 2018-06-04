#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

const std::string base = "videos/";
const std::string folder = base + "new";
const int skip_frames = 220;
const bool wrapAround = true;
const bool recalibrate = false;
const bool enable_local = true;
const bool save_video = false;
const bool show_out = true;
const bool use_stream = true;
const bool debug_stream = true;
const bool use_surf = true;
const int NUM_IMAGES = 1;
//const int offsets[NUM_IMAGES] = {0, 37, 72, 72, 37}; // static
//const int offsets[NUM_IMAGES] = {43, 153, 151, 131, 95, 0}; // new
const int offsets[NUM_IMAGES] = { 0 }; // dynamic
const int RECALIB_DEL = 2;
const double WORK_MEGAPIX = 0.6;	//0.6;	//-1			// Megapix parameter is scaled to the number
const double SEAM_MEAGPIX = 0.01;							// of pixels in full image and this is used
const double COMPOSE_MEGAPIX = 1.4;	//1.4;	//2.2;	//-1	// as a scaling factor when resizing images
const float MATCH_CONF = 0.5f;
const float BLEND_STRENGTH = 5;
const int HESS_THRESH = 300;
const int NOCTAVES = 4;
const int NOCTAVESLAYERS = 2;

const int MAX_FEATURES_PER_IMAGE = 20;
const bool VISUALIZE_MATCHES = false; // Draw the meshes and matches to images pre mesh warp
const bool VISUALIZE_WARPED = false; // Draw the warped mesh
const int N = 2;
const int M = 2;
// Alphas are weights for different cost functions
// 0: Local alignment, 1: Global alignment, 2: Smoothness
const float ALPHAS[3] = {1, 0.11, 0.001};
const int GLOBAL_DIST = 5; // Maximum distance from vertex in global warping

// Test material before right videos are obtained from the camera rig
const std::vector<std::string> video_files = {folder + "/0.mp4", folder + "/1.mp4", folder + "/2.mp4", folder + "/3.mp4", folder + "/4.mp4", folder + "/5.mp4"};

#define PI 3.1415926535897932384626
#define LOGLN(msg) std::cout << msg << std::endl

// Comment this out when on windows
//#define LINUX
