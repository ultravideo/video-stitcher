#include "networking.h"
#include <WS2tcpip.h>
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>

#define DEFAULT_BUFLEN 1024
#define DEFAULT_PORT "6666"
#define DEFAULT_ADDRESS NULL

int startPolling(std::vector<BlockingQueue<cv::Mat>> &queue, std::thread &th)
{
	WSADATA wsaData;
	SOCKET ConnectSocket = INVALID_SOCKET;
	struct addrinfo *result = NULL,
		*ptr = NULL,
		hints;
	char *sendbuf = "this is a test";
	int iResult;

	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		printf("WSAStartup failed with error: %d\n", iResult);
		return 1;
	}

	ZeroMemory(&hints, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	// Resolve the server address and port
	iResult = getaddrinfo(DEFAULT_ADDRESS, DEFAULT_PORT, &hints, &result);
	if (iResult != 0) {
		printf("getaddrinfo failed with error: %d\n", iResult);
		WSACleanup();
		return 1;
	}

	// Attempt to connect to an address until one succeeds
	for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {

		// Create a SOCKET for connecting to server
		ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype,
			ptr->ai_protocol);
		if (ConnectSocket == INVALID_SOCKET) {
			printf("socket failed with error: %ld\n", WSAGetLastError());
			WSACleanup();
			return 1;
		}

		// Connect to server.
		iResult = connect(ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
		if (iResult == SOCKET_ERROR) {
			closesocket(ConnectSocket);
			ConnectSocket = INVALID_SOCKET;
			continue;
		}
		break;
	}

	freeaddrinfo(result);

	if (ConnectSocket == INVALID_SOCKET) {
		printf("Unable to connect to server!\n");
		WSACleanup();
		return 1;
	}

	// shutdown the connection since no more data will be sent
	//iResult = shutdown(ConnectSocket, SD_SEND);
	//if (iResult == SOCKET_ERROR) {
	//	printf("shutdown failed with error: %d\n", WSAGetLastError());
	//	closesocket(ConnectSocket);
	//	WSACleanup();
	//	return 1;
	//}

	th = std::thread(pollFrames, std::ref(ConnectSocket), std::ref(queue));
	return 0;
}

#define IMG_WIDTH 1280
#define IMG_HEIGHT 720
#define CHANNELS 3
void pollFrames(SOCKET &ConnectSocket, std::vector<BlockingQueue<cv::Mat>> &queue)
{
	int num_imgs = queue.size();
	int iResult;
	const int recvbuflen = DEFAULT_BUFLEN;
	char recvbuf[recvbuflen];

	cv::Mat mat(cv::Size(IMG_WIDTH, IMG_HEIGHT), CV_8U);
	int index = 0;
	int img_idx = 0;
	// Receive until the peer closes the connection
	do {
		iResult = recv(ConnectSocket, recvbuf, recvbuflen, 0);
		if (iResult > 0) {
			printf("Bytes received: %d\n", iResult);
			memcpy(mat.data + index, recvbuf, iResult);
		}
		else if (iResult == 0) {
			printf("Connection closed\n");
		}
		else {
			printf("recv failed with error: %d\n", WSAGetLastError());
		}

		index += iResult;
		if (index >= IMG_WIDTH * IMG_HEIGHT * CHANNELS) {
			index = 0;
			queue[img_idx].push(mat);
			++img_idx;
			mat = cv::Mat();
		}
		if (img_idx == num_imgs) {
			img_idx = 0;
		}

	} while (1);

	// cleanup
	closesocket(ConnectSocket);
	WSACleanup();
}