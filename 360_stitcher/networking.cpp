#include "networking.h"
#include "defs.h"
#include "netlib.h"
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <iostream>
#include <fcntl.h>

static const int MAX_CLIENTS = 10;
static const int DEFAULT_BUFLEN = 10;

/* initialized in startPolling() */
static sts_net_socket_t server_socket;

static void pollFrames(sts_net_socket_t *socket, BlockingQueue<cv::Mat> &queue)
{
    int max_idx = CAPTURE_IMG_HEIGHT * CAPTURE_IMG_WIDTH * CAPTURE_IMG_CHANNELS;
    char recvbuf[DEFAULT_BUFLEN];
    int copy_length, overflow, index, error_count;

    cv::Mat mat(cv::Size(CAPTURE_IMG_WIDTH, CAPTURE_IMG_HEIGHT), CV_MAKETYPE(CV_8U, CAPTURE_IMG_CHANNELS));
    index = error_count = 0;

    /* Receive until peer closes the connection */
    do {
        memset(recvbuf, 0, DEFAULT_BUFLEN);
        int bytes = sts_net_recv(socket, recvbuf, sizeof(recvbuf) - 1);

        if (bytes < 0) {
            LOGLN("ERROR: recv failed");
            if (++error_count > 3) {
                LOGLN("ERROR: maximum number of errors reached!");
                break;
            }
        } else if (bytes == 0) {
            LOGLN("connection was closed");
            break;
        } else {
            copy_length = cv::min(bytes, max_idx - index);
            memcpy(mat.data + index, recvbuf, copy_length);
            index += bytes;
        }

        if (index >= CAPTURE_IMG_WIDTH * CAPTURE_IMG_HEIGHT * CAPTURE_IMG_CHANNELS) {
            index = 0;
            cv::cvtColor(mat, mat, CV_YUV2BGR_NV12);
            queue.push(mat);

            mat = cv::Mat(cv::Size(CAPTURE_IMG_WIDTH, CAPTURE_IMG_HEIGHT), CV_MAKETYPE(CV_8U, CAPTURE_IMG_CHANNELS));
            overflow = bytes - copy_length;

            if (overflow) {
                memcpy(mat.data, recvbuf + copy_length, overflow);
                index = overflow;
            }
        }
    } while (1);

    sts_net_close_socket(socket);
    sts_net_shutdown();
}

static void pollClients(sts_net_socket_t *socket, std::vector<BlockingQueue<cv::Mat>> &queue)
{
    int ptr = 0;
    int ConnectSocket;
    sts_net_socket_t clients[MAX_CLIENTS];

    LOGLN("listening...");

    while (1) {
        if (sts_net_accept_socket(socket, &clients[ptr]) < 0) {
            fprintf(stderr, "ERROR: accept failed: '%s'\n", sts_net_get_last_error());
            continue;
        } else {
            LOGLN("client connected");
            std::thread th(pollFrames, &clients[ptr], std::ref(queue[ptr]));
            th.detach();

            if (++ptr == MAX_CLIENTS) {
                while (1) {
                    LOGLN("ERROR: maximum number of clients reached");
                    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
                }
            }
        }
    }

    sts_net_close_socket(socket);
}

int startPolling(std::vector<BlockingQueue<cv::Mat>> &queues)
{
    sts_net_init();

    if (sts_net_open_socket(&server_socket, NULL, CAPTURE_TCP_PORT) < 0) {
        fprintf(stderr, "ERROR: failed to open server socket: %s\n", sts_net_get_last_error());
        return 1;
    }

    std::thread th = std::thread(pollClients, &server_socket, std::ref(queues));
    th.detach();

    return 0;
}