////////////////////////////////////////////////////////////////////////////////
/*
sts_net.h - v0.07 - public domain
written 2017 by Sebastian Steinhauer

LICENSE
Public domain. See "unlicense" statement at the end of this file.

ABOUT
A simple BSD socket wrapper.

REMARKS
The packet API is still work in progress.

*/
////////////////////////////////////////////////////////////////////////////////
#ifndef __INCLUDED__STS_NET_H__
#define __INCLUDED__STS_NET_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifndef STS_NET_SET_SOCKETS
// define a bigger default if needed
// this is the maximum amount of sockets you can keep in a socket set
#define STS_NET_SET_SOCKETS   32
#endif // STS_NET_SET_SOCKETS

#ifndef STS_NET_BACKLOG
// amount of waiting connections for a server socket
#define STS_NET_BACKLOG       2
#endif // STS_NET_BACKLOG

#ifndef STS_NET_NO_PACKETS
#ifndef STS_NET_PACKET_SIZE
// the biggest possible size for a packet
// note, that this size is already bigger then any MTU
#define STS_NET_PACKET_SIZE   2048
#endif // STS_NET_PACKET_SIZE
#endif // STS_NET_NO_PACKETS


////////////////////////////////////////////////////////////////////////////////
//
//    Structures
//
typedef struct {
	int   fd;             // socket file descriptor
	int   ready;          // flag if this socket is ready or not
	int   server;         // flag indicating if it is a server socket
#ifndef STS_NET_NO_PACKETS
	int   received;       // number of bytes currently received
	int   packet_length;  // the packet size which is requested (-1 if it is still receiving the first 2 bytes)
	char  data[STS_NET_PACKET_SIZE];  // buffer for the incoming packet
#endif // STS_NET_NO_PACKETS
} sts_net_socket_t;


typedef struct {
	sts_net_socket_t *sockets[STS_NET_SET_SOCKETS];
} sts_net_set_t;


// REMARK: most functions return 0 on success and -1 on error. You can get a more verbose error message
// from sts_net_get_last_error. Functions which behave differently are the sts_net packet api and sts_net_check_socket_set.


////////////////////////////////////////////////////////////////////////////////
//
//    General API
//
// Get the last error from sts_net (can be called even before sts_net_init)
const char *sts_net_get_last_error();

// Initialized the sts_net library. You have to call this before any other function (except sts_net_get_last_error)
int sts_net_init();

// Shutdown the sts_net library.
void sts_net_shutdown();


////////////////////////////////////////////////////////////////////////////////
//
//    Low-Level Socket API
//
// Reset a socket (clears the structure).
// THIS WILL NOT CLOSE the socket. It's ment to "clear" the socket structure.
void sts_net_reset_socket(sts_net_socket_t *socket);

// Check if the socket structure contains a "valid" socket.
int sts_net_is_socket_valid(sts_net_socket_t *socket);

// Open a (TCP) socket. If you provide "host" sts_net will try to connect to a remove host.
// Pass NULL for host and you'll have a server socket.
int sts_net_open_socket(sts_net_socket_t *socket, const char *host, const char *service);

// Closes the socket.
void sts_net_close_socket(sts_net_socket_t *socket);

// Try to accept a connection from the given server socket.
int sts_net_accept_socket(sts_net_socket_t *listen_socket, sts_net_socket_t *remote_socket);

// Send data to the socket.
int sts_net_send(sts_net_socket_t *socket, const void *data, int length);

// Receive data from the socket.
// NOTE: this call will block if the socket is not ready (meaning there's no data to receive).
int sts_net_recv(sts_net_socket_t *socket, void *data, int length, int flags);

// Initialized a socket set.
void sts_net_init_socket_set(sts_net_set_t *set);

// Add a socket to the socket set.
int sts_net_add_socket_to_set(sts_net_socket_t *socket, sts_net_set_t *set);

// Remove a socket from the socket set. You have to remove the socket from a set manually.
// sts_net_close_socket WILL NOT DO THAT!
int sts_net_remove_socket_from_set(sts_net_socket_t *socket, sts_net_set_t *set);

// Checks for activity on all sockets in the given socket set. If you want to peek for events
// pass 0.0f to the timeout.
// All sockets will have set the ready property to non-zero if you can read data from it,
// or can accept connections.
//  returns:
//    -1  on errors
//     0  if there was no activity
//    >0  amount of sockets with activity
int sts_net_check_socket_set(sts_net_set_t *set, const float timeout);


#ifndef STS_NET_NO_PACKETS
// try to "refill" the internal packet buffer with data
// note that the socket has to be "ready" so use it in conjunction with a socket set
// returns:
//  -1  on errors
//   0  if there was no data
//   1  added some bytes of new packet data
int sts_net_refill_packet_data(sts_net_socket_t *socket);

// tries to "decode" the next packet in the stream
// returns 0 when there's no packet read, non-zero if you can use socket->data and socket->packet_length
int sts_net_receive_packet(sts_net_socket_t *socket);

// drops the packet after you used it
void sts_net_drop_packet(sts_net_socket_t *socket);
#endif // STS_NET_NO_PACKETS
#endif // __INCLUDED__STS_NET_H__


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////
////    IMPLEMENTATION
////
////

//// On Windows 64-bit, almost all socket functions use the type SOCKET
//// to operate, but it's safe to cast it down to int, because handles
//// can't get bigger then 2^24 on Windows...so I don't know why SOCKET is 2^64 ;)
//// https://msdn.microsoft.com/en-us/library/ms724485(VS.85).aspx

#include <string.h>   // NULL and possibly memcpy, memset

#ifdef _WIN32
#include <WinSock2.h>
#include <Ws2tcpip.h>
typedef int socklen_t;
#pragma comment(lib, "Ws2_32.lib")
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#define INVALID_SOCKET    -1
#define SOCKET_ERROR      -1
#define closesocket(fd)   close(fd)
#endif


#ifndef sts__memcpy
#define sts__memcpy     memcpy
#endif // sts__memcpy
#ifndef sts__memset
#define sts__memset     memset
#endif // sts__memset

static const char *sts_net__error_message = "";

#ifdef __cplusplus
}
#endif
/*
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
*/