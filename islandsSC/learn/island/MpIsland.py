"""
@title

@description

"""
import json
import logging
import socket
import threading
import time
from json import JSONDecodeError

from island_influence.learn.island.MAIsland import MAIsland
from island_influence.learn.island.ThreadIsland import ThreadIsland


class MpIsland(ThreadIsland):

    def __init__(self, agent_populations, evolving_agent_names, env, optimizer, max_iters, save_dir, migrate_every=1,
                 name=None, track_progress=False, logger=None, host='127.0.0.1', server_port=0):
        super().__init__(agent_populations, evolving_agent_names, env, optimizer, max_iters, save_dir, migrate_every=migrate_every,
                         name=name, track_progress=track_progress, logger=logger)
        self.accept_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        # server socket is only ever used to accept new connections
        # each connection accepted by the server socket starts its socket with a thread that listens for, and sends, messages
        self.server_port = server_port

        self.socket_timeout = 1
        self.connection_dead_time = 60
        self._listening = False
        self.neighbors = {}

        self._listen_connection_thread = threading.Thread(target=self._listen_connections, daemon=True)
        self._listen_neighbors_thread = threading.Thread(target=self._listen_neighbors, daemon=True)
        # todo  fix rollouts are significantly slower when using the mp module
        #       because the entire rollout is moved to another thread?
        return

    def debug_message(self, message):
        print(f'[{self.name}]: {message}')
        return

    def connect(self, host, port):
        new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        new_socket.connect((host, port))
        new_socket.send(self.name.encode('utf-8'))
        island_id = new_socket.recv(1024)
        island_id = island_id.decode('utf-8')
        self.add_neighbor((island_id, new_socket))
        return

    def close(self):
        super().close()
        self.accept_socket.close()
        return

    def start_listeners(self):
        self._listening = True
        self._listen_connection_thread.start()
        self._listen_neighbors_thread.start()
        return

    def send_data(self, data, recipients=None):
        if recipients is None:
            recipients = list(self.neighbors.keys())

        # todo  RuntimeError: dictionary changed size during iteration
        for island_id, neighbor in self.neighbors.items():
            connection = neighbor['connection']
            if island_id in recipients:
                try:
                    if isinstance(data, str):
                        data = dict(msg_type='data', msg=data, time=time.time())
                    if isinstance(data, dict):
                        data = json.dumps(data)
                    data = data.encode('utf-8')
                    connection.sendall(data)
                except Exception as e:
                    self.debug_message(f'Failed to send data. Error: {e}')
        return

    @staticmethod
    def format_message(message_type, message):
        formatted_msg = dict(msg_type=message_type, msg=message, time=time.time())
        return formatted_msg

    def _handle_neighbor(self, island_id):
        heartbeat = HeartBeat(parent=self, island_name=island_id, beat_interval=5)
        heartbeat.run()
        client_socket = self.neighbors[island_id]
        client_socket = client_socket['connection']
        client_socket.settimeout(self.socket_timeout)
        self.neighbors[island_id]['alive'] = True
        self.debug_message(f'Handling neighbor connection: {client_socket=} | {client_socket.gettimeout()}')
        while self._listening and self.neighbors[island_id]['alive']:
            try:
                data = client_socket.recv(4096)
                if not data:
                    # client has closed the connection
                    break

                data = data.decode('utf-8')
                try:
                    data = json.loads(data)
                    self._handle_message(island_id, data)
                except JSONDecodeError as jde:
                    self.debug_message(f'Received a malformed message from {island_id}: {client_socket}')
            except socket.timeout as te:
                time_since_last = time.time() - self.neighbors[island_id]['last_message']
                self.debug_message(f'{time_since_last=}')
                if time_since_last > self.connection_dead_time:
                    self.neighbors[island_id]['alive'] = False
            except ConnectionResetError as cre:
                self.debug_message(f'Socket has been closed remotely: {cre}')
                break
        self.debug_message(f'Closing socket: {client_socket}')
        client_socket.close()
        heartbeat.stop()
        self.neighbors.pop(island_id)
        return

    def _listen_connections(self):
        self.accept_socket.bind((self.host, self.server_port))
        self.port = self.accept_socket.getsockname()[1]
        self.accept_socket.listen()
        self.debug_message(f'Listening for connections on {self.host}:{self.port}')

        while self._listening:
            try:
                conn, raddr = self.accept_socket.accept()
                island_id = conn.recv(2048).decode('utf-8')
                conn.send(self.name.encode('utf-8'))
                self.add_neighbor((island_id, conn))
            except socket.timeout as te:
                # self.debug_message('No connection attempts')
                pass
        return

    def _listen_neighbors(self):
        while self._listening:
            pass
        self.close()
        return

    def _handle_message(self, sender, message):
        msg_type = message['msg_type']
        self.neighbors[sender]['last_message'] = time.time()
        if msg_type == 'keep_alive':
            return
        self.debug_message(f'Received message from {sender}: {message}')
        return

    def remove_neighbor(self, neighbor_id):
        success = False
        if neighbor_id in self.neighbors:
            self.neighbors.pop(neighbor_id)
            # need to stop thread associated with this island
            self.neighbors[neighbor_id]['alive'] = False
            self.debug_message(f'Removed island {neighbor_id} from known neighbors')
            success = True
        return success

    def add_neighbor(self, neighbor):
        # todo  handle case where the island_id already exists, but it's associated with a different address
        island_id, connection = neighbor
        if island_id == self.name:
            self.debug_message(f'Refusing connection attempt received from self')
        elif island_id not in self.neighbors.keys():
            self.debug_message(f'Accepted connection from {island_id}: {connection}')
            client_thread = threading.Thread(target=self._handle_neighbor, args=(island_id,), daemon=True)
            self.neighbors[island_id] = {'connection': connection, 'alive': True, 'last_message': time.time()}
            client_thread.start()
        else:
            self.debug_message(f'{island_id} already in list on connections')
        return

    def receive_population(self, pop_id, population, from_neighbor):
        return

    def send_populations(self):
        for each_neighbor in self.neighbors:
            for agent_type, population in self.agent_populations.items():
                if agent_type in self.evolving_agent_names:
                    num_agents = each_neighbor.env.num_agent_types(agent_type)
                    self.sort_population(agent_type)
                    top_agents = self.agent_populations[agent_type]
                    top_agents = top_agents[:num_agents]

                    logging.debug(f'Migration: {time.time()}: {len(top_agents)} agents: {self.name} -> {each_neighbor.name}')
                    if isinstance(each_neighbor, MpIsland):
                        pass
                        # todo  need to handle island on a different process differently than islands on the same process
                    else:
                        each_neighbor.receive_population(agent_type, top_agents, self)
        return


class HeartBeat:

    def __init__(self, parent, island_name, beat_interval=5):
        self.alive = False
        self.parent = parent
        self.island_name = island_name
        self.beat_interval = beat_interval
        self.beat_thread = threading.Thread(target=self.beat, daemon=True)
        return

    def stop(self):
        self.parent.neighbors[self.island_name]['alive'] = False
        return

    def run(self):
        self.beat_thread.start()
        return

    def beat(self):
        keep_alive_message = {'msg_type': 'keep_alive', 'time': time.time()}
        self.parent.debug_message(f'Heartbeat started')
        while self.parent.neighbors[self.island_name]['alive']:
            self.parent.send_data(keep_alive_message, [self.island_name])
            time.sleep(self.beat_interval)
        self.parent.debug_message(f'Heartbeat stopped')
        return
