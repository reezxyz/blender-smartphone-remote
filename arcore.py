# ##### BEGIN GPL LICENSE BLOCK #####
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ##### END GPL LICENSE BLOCK #####

import zmq
import msgpack
import numpy as np

import queue
import threading
import logging
import time

from . import environment

TIMEOUT = 1

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Camera():
    """ ARCore camera

    """
    def __init__(self,intrinsics=[0,0], vm=None):

        self.intrinsics = intrinsics
        if vm:
            self.view_matrix =  np.matrix([[vm[0],vm[1],vm[2],vm[3]],
                                        [vm[4],vm[5],vm[6],vm[7]],
                                        [vm[8],vm[9],vm[10],vm[11]],
                                        [vm[12],vm[13],vm[14],vm[15]]])
        else:
            self.view_matrix = np.matrix(np.zeros((4, 4)))


class Node():
    """Sceneform Node

    """

    def __init__(self, wm = None):
        if wm:
            self.world_matrix =  np.matrix([[wm[0],wm[1],wm[2],wm[3]],
                                        [wm[4],wm[5],wm[6],wm[7]],
                                        [wm[8],wm[9],wm[10],wm[11]],
                                        [wm[12],wm[13],wm[14],wm[15]]])
        else:
            self.world_matrix = np.matrix(np.zeros((4, 4)))


class Frame():
    """ ARCore frame

    """
    def __init__(self, camera = Camera(),root = Node()):
        self.camera = camera
        self.root = root
        self.mode = 'CAMERA'

    @classmethod
    def recv(cls,socket):
        frame_buffer = socket.recv_multipart()
        frame = Frame()

        while len(frame_buffer)>0:
            header =  frame_buffer.pop(0)
            if header == b"STATE":
                frame.mode = frame_buffer.pop(0).decode()

            elif header == b"CAMERA":
                arCamera = Camera(
                    intrinsics=msgpack.unpackb(frame_buffer.pop(0)),
                    vm=msgpack.unpackb(frame_buffer.pop(0))
                    )

                frame.camera = arCamera
            elif header == b"NODE":
                arNode = Node(
                    wm=msgpack.unpackb(frame_buffer.pop(0))
                    )
                frame.root = arNode

        return frame


class ArEventHandler():
    """ Application Event handler

    Used to link Android Application event with
    the local application(ex: blender).
    """
    def __init__(self):
        self._frameListeners = []
        self._getScene = None
        self._record = None
        self._ping = None

    def bindPing(self, f):
        self._ping = f

    def OnPing(self, client_addr):
        if self._ping:
            self._ping(client_addr)

    def bindOnFrameReceived(self,f):
        self._frameListeners.append(f)

    def bindGetScene(self, f):
        self._getScene = f

    def bindRecord(self, f):
        self._record = f

    def OnFrameReceived(self, frame):
        for f in self._frameListeners:
            f(frame)

    def OnRecord(self, status):
        if self._record:
            return self._record(status)
        else:
            log.info("Record not implemented")
            return None

    def OnGetScene(self,offset,chunk_size):
        if self._getScene:
            return self._getScene(offset,chunk_size)
        else:
            raise Exception('GetScene not implemented !')


class ArCoreInterface(object):
    """ Application Ar interface

    Interface in charge to manage the network bridge
    between local application and the remote app.
    """
    def __init__(self, handler, port=63150):
        self._net_link = AppLink(handler, port=port)
        self.handler = handler

    def start(self):
        self._net_link.start()

    def is_running(self):
        return self._net_link.is_alive()

    def stop(self):
        self._net_link.stop()


class AppLink(threading.Thread):
    """ Application network link

    Thread use to listen android remote application.

    The network loop (frames, heartbeat, record) never blocks on a scene
    export: SCENE requests are handled by a worker thread and the answers
    are sent back by the loop itself (zmq sockets are not thread safe).
    """
    def __init__(
            self,
            handler,
            port=63150,
            context=None,
            name="Applink"):

        threading.Thread.__init__(self)
        self.name = name
        self.daemon = True  # never keep Blender alive after quit
        self.status = 0  # 0: idle, 1: Connecting, 2: Connected
        self.context = context or zmq.Context.instance()

        self.command_socket = None
        self.data_socket = None
        self.ttl_socket = None

        try:
            # The command socket handle asynchonous commands
            # the from android application
            self.command_socket = self.context.socket(zmq.ROUTER)
            self.command_socket.setsockopt(zmq.IDENTITY, b'SERVER')
            self.command_socket.setsockopt(zmq.RCVHWM, 0)
            self.command_socket.linger = 0
            self.command_socket.bind("tcp://*:{}".format(port+2))

            # Data socket is used load each frame ARCore state from
            # the android application
            self.data_socket = self.context.socket(zmq.PULL)
            self.data_socket.linger = 0
            self.data_socket.bind("tcp://*:{}".format(port+1))

            # TTL socket is in charge of the connexion monitoring
            self.ttl_socket = self.context.socket(zmq.ROUTER)
            self.ttl_socket.setsockopt(zmq.IDENTITY, b'SERVERTTL')
            self.ttl_socket.setsockopt(zmq.RCVHWM, 0)
            self.ttl_socket.linger = 0
            self.ttl_socket.bind("tcp://*:{}".format(port))
        except Exception:
            # Do not leak half-bound ports: otherwise every following
            # Start fails with "Address already in use" until Blender restarts.
            self._close_sockets()
            raise

        self.client_addr = None

        self.exit_event = threading.Event()
        self.handler = handler

        self._scene_requests = queue.Queue()
        self._outbox = queue.Queue()

    def _close_sockets(self):
        for name in ("data_socket", "ttl_socket", "command_socket"):
            sock = getattr(self, name, None)
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
                setattr(self, name, None)

    def _scene_worker(self):
        """Compute scene chunks away from the network loop."""
        while not self.exit_event.is_set():
            try:
                identity, offset, chunk_size = self._scene_requests.get(
                    timeout=0.2)
            except queue.Empty:
                continue

            try:
                chunk = self.handler.OnGetScene(offset, chunk_size) or b""
            except Exception:
                log.exception("GetScene failed")
                chunk = b""

            # Always answer: a missing answer leaves the phone waiting
            # forever ("nothing happens").
            self._outbox.put((identity, [b"SCENE", chunk]))

    def _flush_outbox(self):
        while True:
            try:
                identity, payload = self._outbox.get_nowait()
            except queue.Empty:
                return
            self.command_socket.send(identity, zmq.SNDMORE)
            self.command_socket.send_multipart(payload)

    def run(self):
        log.debug("Running App link")

        poller = zmq.Poller()
        poller.register(self.data_socket, zmq.POLLIN)
        poller.register(self.ttl_socket, zmq.POLLIN)
        poller.register(self.command_socket, zmq.POLLIN)

        worker = threading.Thread(
            target=self._scene_worker, name="ApplinkScene", daemon=True)
        worker.start()

        try:
            while not self.exit_event.is_set():
                try:
                    items = dict(poller.poll(TIMEOUT))

                    # Handle arcore data replication
                    if self.data_socket in items:
                        try:
                            self.handler.OnFrameReceived(
                                Frame.recv(self.data_socket))
                        except Exception as e:
                            log.warning("bad frame: {}".format(e))

                    # Handle live link heartbeat
                    if self.ttl_socket in items:
                        request = self.ttl_socket.recv_multipart()
                        identity = request[0]

                        if self.client_addr is None and len(request) > 2:
                            self.client_addr = request[2].decode()

                        try:
                            self.handler.OnPing(self.client_addr)
                        except Exception as e:
                            log.warning("ping hook failed: {}".format(e))

                        self.ttl_socket.send(identity, zmq.SNDMORE)
                        self.ttl_socket.send_multipart([b"pong"])

                    # Handle remote command
                    if self.command_socket in items:
                        command = self.command_socket.recv_multipart()
                        identity = command[0]

                        if command[1] == b"SCENE":
                            log.debug("Scene requested: offset={} size={}"
                                      .format(int(command[2]),
                                              int(command[3])))
                            self._scene_requests.put(
                                (identity, int(command[2]), int(command[3])))

                        elif command[1] == b"RECORD":
                            result = self.handler.OnRecord(
                                command[2].decode())
                            response = [b"RECORD"]
                            if result:
                                response.append(result.encode())
                            else:
                                response.append(b"STOPPED")

                            self.command_socket.send(identity, zmq.SNDMORE)
                            self.command_socket.send_multipart(response)
                        else:
                            log.error("Wrong request")

                    self._flush_outbox()

                except zmq.ZMQError as e:
                    if self.exit_event.is_set():
                        break
                    log.error("zmq error: {}".format(e))
                except Exception:
                    # One bad message must never kill the link thread
                    log.exception("App link loop error")
        finally:
            log.debug("Exiting App link")
            self._close_sockets()

    def stop(self):
        self.exit_event.set()




if __name__ == "__main__":
    ar_handler = ArEventHandler()

    def test(frame):
        log.debug(frame.camera.view_matrix)

    ar_handler.bindOnFrameReceived(test)

    ar = ArCoreInterface(ar_handler)

    ar.start()
    time.sleep(5)


    ar.stop()
