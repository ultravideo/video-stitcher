#pragma once

#include "defs.h"
#include "blockingqueue.h"
#include <vector>
#include <opencv2/core.hpp>

int startPolling(std::vector<BlockingQueue<cv::Mat>> &queue);