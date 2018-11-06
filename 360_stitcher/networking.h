#pragma once
#include "defs.h"
#include "blockingqueue.h"
#include <vector>
#include <opencv2/core.hpp>


#ifndef LINUX
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")
#endif

int startPolling(std::vector<BlockingQueue<cv::Mat>> &queue);

