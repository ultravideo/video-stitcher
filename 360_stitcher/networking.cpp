#include "networking.h"
#include "defs.h"
#include "debug.h"
#include "netlib.h"
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <iostream>
#include <fcntl.h>

/* initialized in startPolling() */
static sts_net_socket_t server_socket;

static void pollFrames(sts_net_socket_t *socket, BlockingQueue<cv::Mat> &queue, int pos)
{
    const int FRAME_TOTAL_BYTES = CAPTURE_IMG_HEIGHT * CAPTURE_IMG_WIDTH * CAPTURE_IMG_CHANNELS;
    cv::Mat mat(cv::Size(CAPTURE_IMG_WIDTH, CAPTURE_IMG_HEIGHT), CV_MAKETYPE(CV_8U, CAPTURE_IMG_CHANNELS));
    cv::Mat bgr_frame;

    char *recvbuf = new char[128];
    int copy_length, overflow, index, error_count;
    index = error_count = 0;

    /* Receive until peer closes the connection */
    while (1) {
        int bytes = sts_net_recv(socket, recvbuf, 128, 0);

        if (bytes < 0) {
            LOGLN("ERROR: recv failed");
            if (++error_count <= 3)
                continue;
            break;
        } else if (bytes == 0) {
            LOGLN("connection closed");
            break;
        }

        /* recv succeeded, copy recvbuf to mat */
        copy_length = cv::min(bytes, FRAME_TOTAL_BYTES - index);
        memcpy(mat.data + index, recvbuf, copy_length);
        index += bytes;

        /* we got one full image */
        if (index > CAPTURE_IMG_HEIGHT * CAPTURE_IMG_WIDTH * CAPTURE_IMG_CHANNELS) {
            cv::cvtColor(mat, mat, CV_YUV2BGR_NV12);
            queue.push(mat);

            /* sleep a jiffy to give more processing power for kvazaar */
            std::this_thread::sleep_for(std::chrono::milliseconds(80));

            index = 0;
            mat = cv::Mat(cv::Size(CAPTURE_IMG_WIDTH, CAPTURE_IMG_HEIGHT), CV_MAKETYPE(CV_8U, CAPTURE_IMG_CHANNELS));
            overflow = bytes - copy_length;
            
            if (overflow) { 
                memcpy(mat.data, recvbuf + copy_length, overflow);
                index = overflow;
            }
        }
    }

    delete recvbuf;
    sts_net_close_socket(socket);
}

static void pollClients(sts_net_socket_t *socket, std::vector<BlockingQueue<cv::Mat>> &queue)
{
#define __DEBUG__
    int ptr = 0, last_octet = 0;
    sts_net_socket_t clients[NUM_IMAGES], tmp_socket;

    LOGLN("listening...");

    while (1) {
        if ((last_octet = sts_net_accept_socket(socket, &tmp_socket)) < 0) {
            fprintf(stderr, "ERROR: accept failed: '%s'\n", sts_net_get_last_error());
            continue;
        } else {
            LOGLN("client connected");

            last_octet -= clientAddrStart;
#ifdef __DEBUG__
            fprintf(stderr, "%dth client\n", ptr + 1);
            last_octet = ptr;
#endif
            clients[last_octet] = tmp_socket;

            std::thread th(pollFrames, &clients[last_octet], std::ref(queue[last_octet]), ptr);
            th.detach();

            if (++ptr > NUM_IMAGES) {
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