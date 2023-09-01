"""
https://github.com/GianisTsol/python-p2p/tree/master/pythonp2p
https://github.com/GianisTsol/python-p2p/blob/master/pythonp2p/node.py
https://raw.githubusercontent.com/GianisTsol/python-p2p/master/pythonp2p/tests/test_node.py

https://cs.berry.edu/~nhamid/p2p/framework-python.html
https://github.com/robertsdotpm/p2pd

https://github.com/jtriley/pystun
https://github.com/dwoz/python-nat-hole-punching
https://stackoverflow.com/questions/4391872/python-open-a-listening-port-behind-a-router-upnp
https://github.com/flyte/upnpclient
"""
import ipaddress
import json
import socket
import sys
import threading
import time

msg_del_time = 30
PORT = 65432
FILE_PORT = 65433


class NodeConnection(threading.Thread):
    def __init__(self, main_node, sock, node_connection_id, host, port):

        super(NodeConnection, self).__init__()

        self.host = host
        self.port = port
        self.main_node = main_node
        self.sock = sock
        self.terminate_flag = threading.Event()
        self.last_ping = time.time()
        # Variable for parsing the incoming json messages
        self.buffer = ''

        # The id of the connected node
        self.id = node_connection_id

        self.main_node.debug_print(f'NodeConnection.send: Started with client ({self.id}){self.host}:{str(self.port)}')

    def send(self, data):
        try:
            data = data + '-TSN'
            self.sock.sendall(data.encode('utf-8'))

        except Exception as e:
            self.main_node.debug_print(f'NodeConnection.send: Unexpected ercontent/ror: {str(sys.exc_info()[0])}')
            self.main_node.debug_print(f'Exception: {str(e)}')
            self.terminate_flag.set()

    def stop(self):
        self.terminate_flag.set()

    def run(self):
        self.sock.settimeout(10.0)

        while not self.terminate_flag.is_set():
            if time.time() - self.last_ping > self.main_node.dead_time:
                self.terminate_flag.set()
                print(f'node {self.id} is dead')

            line = ''

            try:
                line = self.sock.recv(4096)

            except socket.timeout:
                # self.main_node.debug_print('NodeConnection: timeout')
                pass

            except Exception as e:
                self.terminate_flag.set()
                self.main_node.debug_print('NodeConnection: Socket has been terminated (%s)' % line)
                self.main_node.debug_print(e)

            if line != '':
                try:
                    # BUG: possible buffer overflow when no -TSN is found!
                    self.buffer += str(line.decode('utf-8'))

                except Exception as e:
                    print('NodeConnection: Decoding line error | ' + str(e))

                # Get the messages by finding the message ending -TSN
                index = self.buffer.find('-TSN')
                while index > 0:
                    message = self.buffer[0:index]
                    self.buffer = self.buffer[index + 4::]

                    if message == 'ping':
                        self.last_ping = time.time()
                        # self.main_node.debug_print('ping from ' + self.id)
                    else:
                        self.main_node.node_message(self, message)

                    index = self.buffer.find('-TSN')

            time.sleep(0.01)

        self.main_node.node_disconnected(self)
        self.sock.settimeout(None)
        self.sock.close()
        del self.main_node.nodes_connected[self.main_node.nodes_connected.index(self)]
        time.sleep(1)


class Node(threading.Thread):
    def __init__(self, node_id, host='', port=65432):
        super(Node, self).__init__()

        self.terminate_flag = threading.Event()
        self.pinger = Pinger(self)  # start pinger
        self.debug = True

        self.dead_time = 45  # time to disconnect from node if not pinged, nodes ping after 20s
        self.id = node_id

        self.host = host
        self.ip = host  # own ip, will be changed by connection later
        self.port = port

        self.nodes_connected = []
        self.msgs = {}
        self.peers = []

        hostname = socket.gethostname()
        self.local_ip = socket.gethostbyname(hostname)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.debug_print('Initialisation of the Node on port: ' + str(self.port))
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(10.0)
        self.sock.listen(1)
        return

    def debug_print(self, msg):
        if self.debug:
            print(f'[debug] {self.id}: {str(msg)}')

    def network_send(self, message, exc=None):
        if exc is None:
            exc = []
        for i in self.nodes_connected:
            if i.host in exc:
                pass
            else:
                i.send(json.dumps(message))

    def connect_to(self, host, port=PORT):
        for node in self.nodes_connected:
            if node.host == host:
                self.debug_print(f'[connect_to]: Already connected with this node: {host}:{port}')
                return True

        if not self.check_ip_to_connect(host):
            self.debug_print(f'connect_to: Cannot connect to {host}:{port}')
            return False

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.debug_print('connecting to %s port %s' % (host, port))
            sock.connect((host, port))

            sock.send(self.id.encode('utf-8'))
            connected_node_id = sock.recv(1024).decode('utf-8')

            if self.id == connected_node_id:
                self.debug_print('Possible own ip: ' + host)
                if ipaddress.ip_address(host).is_private:
                    self.local_ip = host
                else:
                    self.ip = host
                self.banned.append(host)
                sock.close()
                return False

            thread_client = self.create_new_connection(sock, connected_node_id, host, port)
            thread_client.start()
            self.nodes_connected.append(thread_client)
            self.node_connected(thread_client)

        except Exception as e:
            self.debug_print('connect_to: Could not connect with node. (' + str(e) + ')')

    def create_new_connection(self, connection, new_conn_id, host, port):
        return NodeConnection(self, connection, new_conn_id, host, port)

    def stop(self):
        self.terminate_flag.set()
        return

    def run(self):
        self.pinger.start()
        while not self.terminate_flag.is_set():  # Check whether the thread needs to be closed
            try:
                connection, client_address = self.sock.accept()
                connected_node_id = connection.recv(2048).decode('utf-8')
                connection.send(self.id.encode('utf-8'))

                if self.id != connected_node_id:
                    thread_client = self.create_new_connection(connection, connected_node_id, client_address[0], client_address[1])
                    thread_client.start()

                    self.nodes_connected.append(thread_client)
                    self.node_connected(thread_client)
                else:
                    connection.close()
            except socket.timeout:
                pass

            except Exception as e:
                raise e

            time.sleep(0.01)

        self.pinger.stop()
        for t in self.nodes_connected:
            t.stop()

        self.sock.close()
        print('Node stopped')

    def connect_to_nodes(self):
        to_remove = []
        for each_peer in self.peers:
            if not self.connect_to(each_peer, PORT):
                to_remove.append(each_peer)
        for each_remove in to_remove:
            self.peers.pop(each_remove)
        return

    def send_message(self, data, receiver=None):
        self.message('msg', data)

    def message(self, msg_type, data):
        # time that the message was sent
        msg_dict = {'type': msg_type, 'data': data, 'time': str(time.time()), 'snid': str(self.id), 'rnid': None}
        self.network_send(msg_dict)

    def send_peers(self):
        self.message('peers', self.peers)

    def data_handler(self, dta, n):
        self.message(dta['type'], dta['data'])
        if len(self.msgs) > len(self.peers) * 20:
            for i in self.msgs.copy():
                if time.time() - self.msgs[i] > msg_del_time:
                    del self.msgs[i]

        data_type = dta['type']
        data = dta['data']

        if data_type == 'peers':
            # peers handling
            for i in data:
                if self.check_ip_to_connect(i):
                    self.peers.append(i)

            self.debug_print(f'Known Peers: {str(self.peers)}')
            self.connect_to_nodes()  # connect to new nodes
            return True

        if data_type == 'msg':
            self.on_message(data, dta['snid'], bool(dta['rnid']))

        if data_type == 'resp':
            self.debug_print('node: ' + dta['snid'] + ' has file ' + data)

    def check_ip_to_connect(self, ip):
        criteria = (
            ip not in self.peers,
            ip != '',
            ip != self.ip,
            ip != self.local_ip,
        )
        return all(criteria)

    def on_message(self, data, sender, private):
        self.debug_print('Incoming Message: ' + data)

    def node_connected(self, node):
        self.debug_print('node_connected: ' + node.id)
        if node.host not in self.peers:
            self.peers.append(node.host)
        self.send_peers()

    def node_disconnected(self, node):
        self.debug_print('node_disconnected: ' + node.id)
        if node.host in self.peers:
            self.peers.remove(node.host)

    def node_message(self, node, data):
        try:
            json.loads(data)
        except json.decoder.JSONDecodeError:
            self.debug_print(f'Error loading message from {node.id}')
            return
        self.data_handler(json.loads(data), [node.host, self.ip])


class Pinger(threading.Thread):
    def __init__(self, parent):
        self.terminate_flag = threading.Event()
        super(Pinger, self).__init__()  # CAll Thread.__init__()
        self.parent = parent
        self.dead_time = 30  # time to disconnect from node if not pinged

    def stop(self):
        self.terminate_flag.set()

    def run(self):
        print('Pinger Started')
        while not self.terminate_flag.is_set():  # Check whether the thread needs to be closed
            for i in self.parent.nodes_connected:
                i.send('ping')
                time.sleep(20)
        print('Pinger stopped')
