#include "networking.h"
#include <WS2tcpip.h>
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/imgproc.hpp>

#define DEFAULT_BUFLEN 1024
#define DEFAULT_PORT "6666"
#define DEFAULT_ADDRESS NULL

int startPolling(std::vector<BlockingQueue<cv::Mat>> &queues)
{
    WSADATA wsaData;
    int iResult;

    SOCKET ListenSocket = INVALID_SOCKET;

    struct addrinfo *result = NULL;
    struct addrinfo hints;

    int iSendResult;
    char recvbuf[DEFAULT_BUFLEN];
    int recvbuflen = DEFAULT_BUFLEN;

    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        printf("WSAStartup failed with error: %d\n", iResult);
        return 1;
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    // Resolve the server address and port
    iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
    if (iResult != 0) {
        printf("getaddrinfo failed with error: %d\n", iResult);
        WSACleanup();
        return 1;
    }

    // Create a SOCKET for connecting to server
    ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (ListenSocket == INVALID_SOCKET) {
        printf("socket failed with error: %ld\n", WSAGetLastError());
        freeaddrinfo(result);
        WSACleanup();
        return 1;
    }

    // Setup the TCP listening socket
    iResult = bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
    if (iResult == SOCKET_ERROR) {
        printf("bind failed with error: %d\n", WSAGetLastError());
        freeaddrinfo(result);
        closesocket(ListenSocket);
        WSACleanup();
        return 1;
    }

    freeaddrinfo(result);

    iResult = listen(ListenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR) {
        printf("listen failed with error: %d\n", WSAGetLastError());
        closesocket(ListenSocket);
        WSACleanup();
        return 1;
    }

	std::thread th = std::thread(pollClients, ListenSocket, std::ref(queues));
    th.detach();
	return 0;
}

#define IMG_WIDTH 1920
#define IMG_HEIGHT 1080
#define CHANNELS 1
void pollFrames(SOCKET ConnectSocket, BlockingQueue<cv::Mat> &queue)
{
	int max_idx = IMG_HEIGHT * IMG_WIDTH * CHANNELS;
	int iResult;
	const int recvbuflen = DEFAULT_BUFLEN;
	char recvbuf[recvbuflen];
	int copy_length;
	int overflow;

	cv::Mat mat(cv::Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC1);
	int index = 0;
	// Receive until the peer closes the connection
	do {
		iResult = recv(ConnectSocket, recvbuf, recvbuflen, 0);
		if (iResult > 0) {
			//printf("Bytes received: %d\n", iResult);
			copy_length = cv::min(iResult, max_idx - index);
			memcpy(mat.data + index, recvbuf, copy_length);
			index += iResult;
		}
		else if (iResult == 0) {
			printf("Connection closed\n");
            break;
		}
		else {
			printf("recv failed with error: %d\n", WSAGetLastError());
		}

		if (index >= IMG_WIDTH * IMG_HEIGHT * CHANNELS) {
			index = 0;
			cv::cvtColor(mat, mat, CV_GRAY2BGR);
			queue.push(mat);

			mat = cv::Mat(cv::Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC1);
			overflow = iResult - copy_length;
			if (overflow) {
				memcpy(mat.data, recvbuf + copy_length, overflow);
				index = overflow;
			}
		}
	} while (1);

	// cleanup
	closesocket(ConnectSocket);
	WSACleanup();
}


void pollClients(SOCKET ListenSocket, std::vector<BlockingQueue<cv::Mat>> &queues)
{
    int idx = 0;
    SOCKET ClientSocket;
    printf("Polling for clients.\n");
    while (1) {
        // Accept a client socket
        ClientSocket = accept(ListenSocket, NULL, NULL);
        if (ClientSocket == INVALID_SOCKET) {
            printf("accept failed with error: %d\n", WSAGetLastError());
        }
        else
        {
            printf("Client connected!");
            std::thread th(pollFrames, std::ref(ClientSocket), std::ref(queues[idx]));
            th.detach();
            ++idx;
            if (idx == queues.size()) {
                idx = 0;
            }
        }
    }
}