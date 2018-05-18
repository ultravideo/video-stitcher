#pragma once
#include <vector>
#include <WinSock2.h>
#include "blockingqueue.h"
#include <opencv2/core.hpp>

#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

int startPolling(std::vector<BlockingQueue<cv::Mat>> &queue);

void pollFrames(SOCKET ConnectSocket, BlockingQueue<cv::Mat> &queue);

void pollClients(SOCKET ListenSocket, std::vector<BlockingQueue<cv::Mat>> &queue);