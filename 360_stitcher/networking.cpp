#include "networking.h"
#ifndef LINUX
#include <WS2tcpip.h>
#include <windows.h>
#include <WinSock2.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/imgproc.hpp>
#include <thread>

#define DEFAULT_BUFLEN 1024
#define DEFAULT_PORT "6666"
#define DEFAULT_ADDRESS NULL


#ifndef LINUX
void pollFrames(SOCKET ConnectSocket, BlockingQueue<cv::Mat> &queue);

void pollClients(SOCKET ListenSocket, std::vector<BlockingQueue<cv::Mat>> &queue);
#else
void pollFrames(int ConnectSocket, BlockingQueue<cv::Mat> &queue);

void pollClients(int ListenSocket, struct sockaddr_in &cli_addr, std::vector<BlockingQueue<cv::Mat>> &queue);
#endif


int startPolling(std::vector<BlockingQueue<cv::Mat>> &queues)
{
#ifdef LINUX
    int sockfd, newsockfd, portno = std::stoi(DEFAULT_PORT);
    socklen_t clilen;
    char buffer[DEFAULT_BUFLEN];
    struct sockaddr_in serv_addr, cli_addr;
    int n;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        printf("ERROR opening socket\n");
        return 1;
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr *) &serv_addr,
             sizeof(serv_addr)) < 0) {
        printf("ERROR on binding\n");
        return 1;
    }

    listen(sockfd, 5);
    clilen = sizeof(cli_addr);
    newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
    if(newsockfd < 0) {
        printf("ERROR on accept\n");
        return 1;
    }
    bzero(buffer, sizeof(buffer));
    n = read(newsockfd, buffer, 255);
    if(n < 0) {
        printf("ERROR reading from socker\n");
        return 1;
    }
    printf("Here is the message: %s\n", buffer);
#else
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
#endif
	return 0;
}

#define IMG_WIDTH 1920
#define IMG_HEIGHT 1080
#define CHANNELS 2

#ifdef LINUX
void pollFrames(int ConnectSocket, BlockingQueue<cv::Mat> &queue)
{
	int max_idx = IMG_HEIGHT * IMG_WIDTH * CHANNELS;
	int iResult;
	const int recvbuflen = DEFAULT_BUFLEN;
	char recvbuf[recvbuflen];
	int copy_length;
	int overflow;

	cv::Mat mat(cv::Size(IMG_WIDTH, IMG_HEIGHT), CV_MAKETYPE(CV_8U, CHANNELS));
	int index = 0;
	// Receive until the peer closes the connection
	do {
        iResult = read(ConnectSocket, recvbuf, recvbuflen);
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
			printf("recv failed\n");
		}

		if (index >= IMG_WIDTH * IMG_HEIGHT * CHANNELS) {
			index = 0;
			cv::cvtColor(mat, mat, CV_YUV2BGRA_YUY2);
			queue.push(mat);

			mat = cv::Mat(cv::Size(IMG_WIDTH, IMG_HEIGHT), CV_MAKETYPE(CV_8U, CHANNELS));
			overflow = iResult - copy_length;
			if (overflow) {
				memcpy(mat.data, recvbuf + copy_length, overflow);
				index = overflow;
			}
		}
	} while (1);

    close(ConnectSocket);
}

void pollClients(int ListenSocket, struct sockaddr_in &cli_addr, std::vector<BlockingQueue<cv::Mat>> &queues)
{
    int idx;
    int ConnectSocket;
    socklen_t clilen = sizeof(cli_addr);
    while(1) {
        ConnectSocket = accept(ListenSocket, (struct sockaddr *) &cli_addr, &clilen);
        if(ConnectSocket < 0) {
            printf("accept failed\n");
        } else {
            std::thread th(pollFrames, ConnectSocket, std::ref(queues[idx]));
            th.detach();
            ++idx;
            if (idx == queues.size()) {
                idx = 0;
            }
        }
    }
    close(ListenSocket);
}
#else
void pollFrames(SOCKET ConnectSocket, BlockingQueue<cv::Mat> &queue)
{
	int max_idx = IMG_HEIGHT * IMG_WIDTH * CHANNELS;
	int iResult;
	const int recvbuflen = DEFAULT_BUFLEN;
	char recvbuf[recvbuflen];
	int copy_length;
	int overflow;

	cv::Mat mat(cv::Size(IMG_WIDTH, IMG_HEIGHT), CV_MAKETYPE(CV_8U, CHANNELS));
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
			cv::cvtColor(mat, mat, CV_YUV2BGRA_YUY2);
			queue.push(mat);

			mat = cv::Mat(cv::Size(IMG_WIDTH, IMG_HEIGHT), CV_MAKETYPE(CV_8U, CHANNELS));
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
#endif
