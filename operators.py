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

import logging
import os
import queue
import threading
import time

import bpy
import numpy as np
from mathutils import Matrix

from .arcore import ArCoreInterface, ArEventHandler
from . import environment, preference, sr_status

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Interval (seconds) of the main-thread executor.
TASKS_FREQUENCY = 0.01
MAX_JOBS_PER_TICK = 32
EXEC_TIMEOUT = 5.0           # normal jobs (camera pose, record state)
EXPORT_TIMEOUT = 180.0       # scene export can be slow on big scenes

# The export freezes Blender's UI. Wait a little before starting so the
# status panel can first show "exporting" on screen.
EXPORT_START_DELAY = 0.4

# Objects with fewer triangles than this are never decimated.
DECIMATE_MIN_TRIS = 300

BLENDER = np.matrix([[-1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])

app = None
is_recording = False

# Each job carries its OWN reply queue, so results can never be mixed up
# between threads.
execution_queue = queue.Queue()

_scene_lock = threading.Lock()
_scene_cache = b""
_scene_cache_time = 0.0
# A request for offset 0 arriving right after an export (e.g. a phone retry
# that queued up while Blender was busy) reuses it instead of exporting again.
SCENE_REUSE_SECONDS = 3.0

# Only the newest AR frame matters: the network thread drops it here and the
# timer applies it. The network loop never waits for Blender.
_frame_lock = threading.Lock()
_pending_frame = None


class SceneTooLarge(Exception):
    """Raised when the scene exceeds the configured limits."""


'''
    Utility functions
'''


def getContext():
    """Return a copy of the current blender context."""
    return bpy.context.copy()


def get_window_override():
    """
    Build a context override with a valid window/screen/area.
    Needed for operators such as screen.animation_play when we are running
    from a timer (where bpy.context.window may be None).
    """
    wm = bpy.context.window_manager
    for window in wm.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                region = next((r for r in area.regions
                               if r.type == 'WINDOW'), None)
                override = dict(window=window, screen=screen, area=area)
                if region:
                    override["region"] = region
                return override
    if wm.windows:
        window = wm.windows[0]
        return dict(window=window, screen=window.screen)
    return {}


'''
    Main thread executor (bpy.app.timers based)

    Network threads call run_in_main_thread(); the job is executed by a
    timer in Blender's main thread. The timer survives exceptions, file
    loads and window changes, and a failing job always answers its caller.
'''


def run_in_main_thread(function, args=None, timeout=EXEC_TIMEOUT, delay=0.0):
    """Run `function` in the main thread and return its result."""
    if threading.current_thread() is threading.main_thread():
        return function(args) if args is not None else function()

    if not bpy.app.timers.is_registered(_executor_tick):
        raise RuntimeError("main thread executor is not running")

    reply = queue.Queue(maxsize=1)
    execution_queue.put((function, args, reply, time.time() + delay))

    try:
        ok, value = reply.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(
            "main thread did not answer within {}s".format(timeout))

    if not ok:
        raise value
    return value


def _executor_tick():
    try:
        _apply_pending_frame()
    except Exception:
        log.exception("applying AR frame failed")

    deferred = []
    for _ in range(MAX_JOBS_PER_TICK):
        try:
            job = execution_queue.get_nowait()
        except queue.Empty:
            break

        function, args, reply, ready_at = job
        if ready_at > time.time():
            deferred.append(job)
            continue

        try:
            result = function(args) if args is not None else function()
            reply.put((True, result))
        except Exception as e:
            if not isinstance(e, SceneTooLarge):
                log.exception("main thread job failed")
            reply.put((False, e))

    for job in deferred:
        execution_queue.put(job)

    return TASKS_FREQUENCY


def start_executor():
    if not bpy.app.timers.is_registered(_executor_tick):
        bpy.app.timers.register(_executor_tick,
                                first_interval=TASKS_FREQUENCY,
                                persistent=True)


def stop_executor():
    global _pending_frame

    if bpy.app.timers.is_registered(_executor_tick):
        bpy.app.timers.unregister(_executor_tick)

    with _frame_lock:
        _pending_frame = None

    # Release anyone still waiting for an answer
    while True:
        try:
            _, _, reply, _ = execution_queue.get_nowait()
        except queue.Empty:
            break
        reply.put((False, RuntimeError("executor stopped")))


'''
    Camera management
'''


def apply_ar_frame(frame):
    """Called from the network thread: never blocks, keeps latest frame."""
    global _pending_frame
    sr_status.count_frame()
    with _frame_lock:
        _pending_frame = frame


def _apply_pending_frame():
    global _pending_frame
    with _frame_lock:
        frame, _pending_frame = _pending_frame, None

    if frame is None:
        return
    if frame.mode == "CAMERA":
        apply_camera_pose(frame)
    elif frame.mode == "OBJECT":
        apply_object_pose(frame)


def _compute_pose(frame):
    bpose = frame.camera.view_matrix * BLENDER
    worigin = frame.root.world_matrix * BLENDER
    bpose[3] = (bpose[3] - worigin[3]) * (1 / np.linalg.norm(worigin[1]))
    return Matrix(bpose.A.tolist())


def apply_object_pose(frame):
    """Apply frame blender pose into blender selected object"""
    try:
        active_object = bpy.context.object
        if active_object:
            active_object.matrix_world = _compute_pose(frame)
    except Exception as e:
        log.error("apply object error: {}".format(e))


def normalize(_d, to_sum=True, copy=True):
    # d is a (n x dimension) np array
    d = _d if not copy else np.copy(_d)
    d -= np.min(d, axis=0)
    d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0))
    return d


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def apply_camera_pose(frame):
    """Apply frame camera pose into blender scene active camera"""
    try:
        camera = bpy.context.scene.camera
        if camera:
            camera.matrix_world = _compute_pose(frame)
            camera.data.angle = frame.camera.intrinsics[0]

            if is_recording:
                camera.keyframe_insert(data_path="location")
                camera.keyframe_insert(data_path="rotation_quaternion")
    except Exception as e:
        log.info("apply camera error: {}".format(e))


def setup_camera_animation():
    global is_recording
    scene = bpy.context.scene

    if scene.camera is None:
        raise RuntimeError("no active camera in the scene")

    # Create and setup new action
    scene.camera.rotation_mode = 'QUATERNION'

    new_action = bpy.data.actions.new("camera_0")
    scene.camera.animation_data_create()
    scene.camera.animation_data.action = new_action

    # Launch record
    override = get_window_override()
    with bpy.context.temp_override(**override):
        screen = bpy.context.screen
        if screen is None or not screen.is_animation_playing:
            bpy.ops.screen.animation_play()
    is_recording = True


def stop_camera_animation():
    global is_recording

    override = get_window_override()
    with bpy.context.temp_override(**override):
        screen = bpy.context.screen
        if screen is not None and screen.is_animation_playing:
            bpy.ops.screen.animation_cancel()
    bpy.context.scene.frame_current = 1
    is_recording = False


def update_camera_animation():
    if is_recording:
        return bpy.context.scene.frame_current
    return "STOPPED"


def record_camera(status):
    """
    Record camera animation callback.

    TODO: Record config
    """
    log.info(status)
    blender_state = "NONE"
    try:
        if status == "START":
            run_in_main_thread(setup_camera_animation)
            blender_state = "STARTED"
        elif status == "STATE":
            blender_state = str(run_in_main_thread(update_camera_animation))
        elif status == "STOP":
            run_in_main_thread(stop_camera_animation)
            blender_state = "STOPPED"
    except Exception as e:
        log.error("record_camera({}) failed: {}".format(status, e))

    return blender_state


'''
   Scene export
'''


def export_cached_scene():
    """
    Export the current scene to SCENE_FILE (runs in the main thread),
    following the export settings of the "Remote > Status & Ekspor" panel.

    Returns the path of the exported file. Raises SceneTooLarge when the
    scene exceeds the configured triangle / file size limits.
    """
    prefs = preference.get()
    cache_dir = prefs.cache_directory
    os.makedirs(cache_dir, exist_ok=True)

    final_path = os.path.join(cache_dir, environment.SCENE_FILE)
    # Export to a temp file first, then swap atomically so a reader can
    # never see a half-written file.
    tmp_path = final_path + ".tmp.obj"

    cfg = sr_status.get_settings()
    limit = cfg.max_triangles
    t0 = time.time()

    # --- 1. how big is the scene? ---------------------------------------
    tris, per_object = sr_status.count_triangles(cfg.apply_modifiers,
                                                 cfg.selected_only)
    tris_before = tris
    tris_after = tris
    use_modifiers = cfg.apply_modifiers
    decimated = False
    temp_modifiers = []
    sr_status.update(objects=len(per_object), tris_before=tris,
                     tris_after=tris, decimated=False)
    log.info("scene: {} objects, {:,} triangles (limit {:,})".format(
        len(per_object), tris, limit))

    try:
        # --- 2. apply the triangle budget -------------------------------
        if limit and tris > limit:
            if cfg.limit_action == 'BLOCK':
                raise SceneTooLarge(
                    "Model terlalu besar: {:,} triangle (batas {:,}). "
                    "Kecilkan model atau naikkan batas.".format(tris, limit))

            # Decimate is a modifier, so modifiers must be evaluated.
            use_modifiers = True
            tris_before, per_object = sr_status.count_triangles(
                True, cfg.selected_only)
            small = sum(t for _, t in per_object if t < DECIMATE_MIN_TRIS)
            big_total = tris_before - small
            tris_after = tris_before

            if big_total > 0 and tris_before > limit:
                ratio = max(0.01, min(1.0, (limit - small) / big_total))
                for obj, t in per_object:
                    if t < DECIMATE_MIN_TRIS:
                        continue
                    try:
                        mod = obj.modifiers.new("_sr_tmp_decimate", 'DECIMATE')
                        mod.decimate_type = 'COLLAPSE'
                        mod.ratio = ratio
                        mod.use_collapse_triangulate = True
                        temp_modifiers.append((obj, mod))
                    except Exception as e:
                        log.warning("cannot decimate {}: {}".format(
                            obj.name, e))
                bpy.context.view_layer.update()
                tris_after = int(small + big_total * ratio)
                decimated = bool(temp_modifiers)

            sr_status.update(tris_before=tris_before, tris_after=tris_after,
                             decimated=decimated)

        # --- 3. export ---------------------------------------------------
        if hasattr(bpy.ops.wm, "obj_export"):
            result = bpy.ops.wm.obj_export(
                filepath=tmp_path,
                check_existing=False,
                export_selected_objects=cfg.selected_only,
                export_animation=False,
                apply_modifiers=use_modifiers,
                export_uv=cfg.export_uv,
                export_normals=cfg.export_normals,
                export_triangulated_mesh=cfg.triangulate,
                export_materials=False,
                forward_axis='NEGATIVE_Z',
                up_axis='Y',
                global_scale=1.0)
        else:
            result = bpy.ops.export_scene.obj(
                filepath=tmp_path,
                check_existing=False,
                use_selection=cfg.selected_only,
                use_animation=False,
                use_mesh_modifiers=use_modifiers,
                use_normals=cfg.export_normals,
                use_uvs=cfg.export_uv,
                use_triangles=cfg.triangulate,
                use_materials=False,
                axis_forward='-Z',
                axis_up='Y')
    finally:
        # Always leave the user's scene exactly as it was
        for obj, mod in temp_modifiers:
            try:
                obj.modifiers.remove(mod)
            except Exception as e:
                log.warning("cannot remove temp modifier on {}: {}".format(
                    obj.name, e))
        if temp_modifiers:
            bpy.context.view_layer.update()

    if 'FINISHED' not in result or not os.path.exists(tmp_path):
        raise RuntimeError("obj export failed: {}".format(result))

    # --- 4. check the resulting size -------------------------------------
    size = os.path.getsize(tmp_path)
    size_mb = size / (1024 * 1024)
    sr_status.update(size_bytes=size, export_seconds=time.time() - t0)

    if cfg.max_size_mb and size_mb > cfg.max_size_mb:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise SceneTooLarge(
            "Hasil ekspor {:.0f} MB melebihi batas {} MB. Turunkan batas "
            "triangle, matikan modifier, atau matikan Normal/UV.".format(
                size_mb, cfg.max_size_mb))

    os.replace(tmp_path, final_path)
    return final_path


def get_cached_scene(offset, chunk_size):
    """
    Return `chunk_size` bytes of the exported scene starting at `offset`.

    The scene is exported once (offset == 0) and the bytes are kept in
    memory, so the whole transfer is consistent even if another request
    starts a new export in the meantime. Progress is published to the
    status panel.
    """
    global _scene_cache, _scene_cache_time

    with _scene_lock:
        fresh = (_scene_cache
                 and time.time() - _scene_cache_time < SCENE_REUSE_SECONDS)

        if offset == 0 and not fresh:
            log.info("export the scene... (chunk_size={})".format(chunk_size))
            sr_status.update(
                phase="EXPORTING",
                message="Blender sedang mengekspor scene. Jendela Blender "
                        "akan membeku sebentar, ini normal.",
                sent_bytes=0, total_bytes=0, size_bytes=0,
                tris_before=0, tris_after=0, decimated=False)
            t0 = time.time()
            try:
                path = run_in_main_thread(export_cached_scene,
                                          timeout=EXPORT_TIMEOUT,
                                          delay=EXPORT_START_DELAY)
                with open(path, "rb") as f:
                    _scene_cache = f.read()
                _scene_cache_time = time.time()
                size_mb = len(_scene_cache) / (1024 * 1024)
                log.info("Scene exported: {:.1f} MB in {:.1f}s".format(
                    size_mb, time.time() - t0))
                sr_status.update(phase="READY", message="",
                                 size_bytes=len(_scene_cache),
                                 total_bytes=len(_scene_cache),
                                 export_seconds=time.time() - t0)
            except SceneTooLarge as e:
                log.warning("scene refused: {}".format(e))
                _scene_cache = b""
                sr_status.update(phase="ERROR", message=str(e))
            except TimeoutError as e:
                log.error("scene export timed out: {}".format(e))
                _scene_cache = b""
                sr_status.update(
                    phase="ERROR",
                    message="Blender tidak selesai mengekspor dalam {:.0f} "
                            "detik. Kecilkan model.".format(EXPORT_TIMEOUT))
            except Exception as e:
                log.error("scene export failed after {:.1f}s: {}".format(
                    time.time() - t0, e))
                _scene_cache = b""
                sr_status.update(phase="ERROR",
                                 message="Ekspor gagal: {}".format(e))

        chunk = _scene_cache[offset:offset + chunk_size]
        total = len(_scene_cache)

        if total:
            sent = min(offset + len(chunk), total)
            if offset == 0:
                sr_status.update(phase="SENDING", sent_bytes=sent,
                                 total_bytes=total,
                                 send_started=time.time())
            elif sent >= total:
                if sr_status.snapshot()["phase"] != "DONE":
                    started = sr_status.snapshot()["send_started"]
                    sr_status.update(phase="DONE", sent_bytes=total,
                                     send_seconds=time.time() - started)
            else:
                sr_status.update(sent_bytes=sent)

        if (not chunk or offset == 0
                or (offset // max(chunk_size, 1)) % 50 == 0):
            log.info("send scene chunk offset={} size={} / total={}".format(
                offset, len(chunk), total))
        return chunk


'''
    Handlers
'''


def record(scene):
    if is_recording and scene.frame_current >= (scene.frame_end - 1):
        stop_camera_animation()


'''
   Operators
'''


class RemoteStartOperator(bpy.types.Operator):
    """Start the blender remote"""
    bl_idname = "remote.start"
    bl_label = "Start remote link"

    def execute(self, context):
        global app

        if app:
            self.report({'WARNING'}, "Remote link is already running")
            return {'CANCELLED'}

        prefs = preference.get()
        handler = ArEventHandler()

        # Arcore interface setup
        handler.bindOnFrameReceived(apply_ar_frame)
        handler.bindGetScene(get_cached_scene)
        handler.bindRecord(record_camera)
        handler.bindPing(sr_status.note_ping)

        # The executor must be up before any client can connect
        start_executor()

        try:
            app = ArCoreInterface(handler, port=prefs.port)
            app.start()
        except Exception as e:
            log.exception("cannot start remote link")
            app = None
            stop_executor()
            self.report({'ERROR'}, "Cannot start remote link: {}".format(e))
            return {'CANCELLED'}

        sr_status.set_server_running(True)

        if record not in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.append(record)

        return {'FINISHED'}


class RemoteStopOperator(bpy.types.Operator):
    """Stop the remote daemon"""
    bl_idname = "remote.stop"
    bl_label = "Stop remote link"

    def execute(self, context):
        global app

        if app:
            try:
                app.stop()
            except Exception as e:
                log.error("error while stopping remote link: {}".format(e))
            app = None

        stop_executor()
        sr_status.set_server_running(False)

        if record in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.remove(record)

        return {'FINISHED'}


class RemoteRestartOperator(bpy.types.Operator):
    """Reboot the remote daemon"""
    bl_idname = "remote.restart"
    bl_label = "Reboot remote link"

    def execute(self, context):
        bpy.ops.remote.stop()
        bpy.ops.remote.start()
        return {'FINISHED'}


class RemoteModalExecutorOperator(bpy.types.Operator):
    """Kept for compatibility: the executor is now a bpy.app.timers job"""
    bl_idname = "wm.modal_executor_operator"
    bl_label = "Modal Executor Operator"

    def execute(self, context):
        start_executor()
        return {'FINISHED'}


classes = (
    RemoteStartOperator,
    RemoteStopOperator,
    RemoteRestartOperator,
    RemoteModalExecutorOperator,
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)

    try:
        sr_status.register()
    except Exception:
        # The status panel is optional: never break the addon because of it
        log.exception("status panel could not be registered")


def unregister():
    from bpy.utils import unregister_class
    global app

    if app:
        try:
            app.stop()
        except Exception as e:
            log.error("error while stopping remote link: {}".format(e))
        app = None

    stop_executor()

    try:
        sr_status.unregister()
    except Exception:
        log.exception("status panel could not be unregistered")

    if record in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(record)

    for cls in reversed(classes):
        unregister_class(cls)
