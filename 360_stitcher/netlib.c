#include "netlib.h"

static int sts_net__set_error(const char *message)
{
	sts_net__error_message = message;
	return -1;
}

void sts_net_reset_socket(sts_net_socket_t *socket)
{
	socket->fd = INVALID_SOCKET;
	socket->ready = 0;
	socket->server = 0;
#ifndef STS_NET_NO_PACKETS
	socket->received = 0;
	socket->packet_length = -1;
#endif // STS_NET_NO_PACKETS
}

int sts_net_is_socket_valid(sts_net_socket_t *socket)
{
	return socket->fd != INVALID_SOCKET;
}

const char *sts_net_get_last_error(void)
{
	return sts_net__error_message;
}

int sts_net_init(void)
{
#ifdef _WIN32
	WSADATA wsa_data;
	if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
		return sts_net__set_error("Windows Sockets failed to start");
	}
	return 0;
#else
	return 0;
#endif // _WIN32
}

void sts_net_shutdown(void)
{
#ifdef _WIN32
	WSACleanup();
#endif // _WIN32
}

int sts_net_open_socket(sts_net_socket_t *sock, const char *host, const char *service)
{
	struct addrinfo     hints;
	struct addrinfo     *res = NULL, *r = NULL;
	int                 fd = INVALID_SOCKET;

	sts_net_reset_socket(sock);
	sts__memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;

	if (host != NULL) {
		// try to connect to remote host
		if (getaddrinfo(host, service, &hints, &res) != 0)
			return sts_net__set_error("Cannot resolve hostname");
		
		for (r = res; r; r = r->ai_next) {
			fd = (int)socket(r->ai_family, r->ai_socktype, r->ai_protocol);
			if (fd == INVALID_SOCKET)
				continue;
			if (connect(fd, r->ai_addr, (int)r->ai_addrlen) == 0)
				break;
			closesocket(fd);
		}

		freeaddrinfo(res);
		if (!r)
			return sts_net__set_error("Cannot connect to host");

		sock->fd = fd;
	} else {
		// listen for connection (start server)
		hints.ai_flags = AI_PASSIVE;
		if (getaddrinfo(NULL, service, &hints, &res) != 0)
			return sts_net__set_error("Cannot resolve hostname");

		fd = (int)socket(res->ai_family, res->ai_socktype, res->ai_protocol);

		if (fd == INVALID_SOCKET) {
			freeaddrinfo(res);
			return sts_net__set_error("Could not create socket");
		}
#ifndef _WIN32
		{
			int yes = 1;
			setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof(yes));
		}
#endif // _WIN32
		if (bind(fd, res->ai_addr, (int)res->ai_addrlen) == SOCKET_ERROR) {
			freeaddrinfo(res);
			closesocket(fd);
			return sts_net__set_error("Could not bind to port");
		}
		freeaddrinfo(res);

		if (listen(fd, STS_NET_BACKLOG) == SOCKET_ERROR) {
			closesocket(fd);
			return sts_net__set_error("Could not listen to socket");
		}
		sock->server = 1;
		sock->fd = fd;
	}
	return 0;
}

void sts_net_close_socket(sts_net_socket_t *socket)
{
	if (socket->fd != INVALID_SOCKET)
		closesocket(socket->fd);
	sts_net_reset_socket(socket);
}

int sts_net_accept_socket(sts_net_socket_t *listen_socket, sts_net_socket_t *remote_socket)
{
	struct sockaddr_in  sock_addr;
	socklen_t           sock_alen;

	if (!listen_socket->server)
		return sts_net__set_error("Cannot accept on client socket");

	if (listen_socket->fd == INVALID_SOCKET)
		return sts_net__set_error("Cannot accept on closed socket");

	sock_alen = sizeof(sock_addr);
	listen_socket->ready = 0;
	remote_socket->ready = 0;
	remote_socket->server = 0;
	remote_socket->fd = (int)accept(listen_socket->fd, (struct sockaddr *)&sock_addr, &sock_alen);
	
	if (remote_socket->fd == INVALID_SOCKET)
		return sts_net__set_error("Accept failed");
	return 0;
}

int sts_net_send(sts_net_socket_t *socket, const void *data, int length)
{
	if (socket->server)
		return sts_net__set_error("Cannot send on server socket");

	if (socket->fd == INVALID_SOCKET)
		return sts_net__set_error("Cannot send on closed socket");

    int bytes_sent = send(socket->fd, (const char *)data, length, 0);

	if (bytes_sent < 0)
		return sts_net__set_error("Cannot send data");

	return bytes_sent;
}

int sts_net_recv(sts_net_socket_t *socket, void *data, int length)
{
	int result;
	if (socket->server)
		return sts_net__set_error("Cannot receive on server socket");
	if (socket->fd == INVALID_SOCKET)
		return sts_net__set_error("Cannot receive on closed socket");

	socket->ready = 0;
	result = recv(socket->fd, (char*)data, length, 0);
	if (result < 0)
		return sts_net__set_error("Cannot receive data");
	return result;
}

void sts_net_init_socket_set(sts_net_set_t *set)
{
	for (int i = 0; i < STS_NET_SET_SOCKETS; ++i) {
		set->sockets[i] = NULL;
	}
}

int sts_net_add_socket_to_set(sts_net_socket_t *socket, sts_net_set_t *set)
{
	if (socket->fd == INVALID_SOCKET)
		return sts_net__set_error("Cannot add closed socket to set");

	for (int i = 0; i < STS_NET_SET_SOCKETS; ++i) {
		if (!set->sockets[i]) {
			set->sockets[i] = socket;
			return 0;
		}
	}
	return sts_net__set_error("Socket set is full");
}

int sts_net_remove_socket_from_set(sts_net_socket_t *socket, sts_net_set_t *set)
{
	if (socket->fd == INVALID_SOCKET)
		return sts_net__set_error("Cannot remove closed socket from set");

	for (int i = 0; i < STS_NET_SET_SOCKETS; ++i) {
		if (set->sockets[i] == socket) {
			set->sockets[i] = NULL;
			return 0;
		}
	}
	return sts_net__set_error("Socket not found in set");
}

int sts_net_check_socket_set(sts_net_set_t *set, const float timeout)
{
	struct timeval  tv;
	fd_set          fds;
	int             i, max_fd, result;

	FD_ZERO(&fds);
	for (i = 0, max_fd = 0; i < STS_NET_SET_SOCKETS; ++i) {
		if (set->sockets[i]) {
			FD_SET(set->sockets[i]->fd, &fds);
			if (set->sockets[i]->fd > max_fd) {
				max_fd = set->sockets[i]->fd;
			}
		}
	}

	if (max_fd == 0)
		return 0;

	tv.tv_sec = (int)timeout;
	tv.tv_usec = (int)((timeout - (float)tv.tv_sec) * 1000000.0f);
	result = select(max_fd + 1, &fds, NULL, NULL, &tv);
	if (result > 0) {
		for (i = 0; i < STS_NET_SET_SOCKETS; ++i) {
			if (set->sockets[i]) {
				if (FD_ISSET(set->sockets[i]->fd, &fds))
					set->sockets[i]->ready = 1;
			}
		}
	} else if (result == SOCKET_ERROR) {
		sts_net__set_error("Error on select()");
	}

	return result;
}


int sts_net_refill_packet_data(sts_net_socket_t *socket)
{
	if (socket->ready)
		return 0;

	int received = sts_net_recv(socket, &socket->data[socket->received], STS_NET_PACKET_SIZE - socket->received);

	if (received < 0)
		return -1;

	socket->received += received;
	return 1;
}

int sts_net_receive_packet(sts_net_socket_t *socket)
{
	if (socket->packet_length < 0) {
		if (socket->received >= 2) {
			socket->packet_length = socket->data[0] * 256 + socket->data[1];
			if (socket->packet_length > STS_NET_PACKET_SIZE) {
				sts_net_close_socket(socket);
				return sts_net__set_error("Received packet was too large");
			}
			socket->received -= 2;
			sts__memcpy(&socket->data[0], &socket->data[2], socket->received);
		}
	}
	return ((socket->packet_length >= 0) && (socket->received >= socket->packet_length));
}


void sts_net_drop_packet(sts_net_socket_t *socket)
{
	if ((socket->packet_length >= 0) && (socket->received >= socket->packet_length)) {
		sts__memcpy(&socket->data[0], &socket->data[socket->packet_length], socket->received - socket->packet_length);
		socket->received -= socket->packet_length;
		socket->packet_length = -1;
	}
}
