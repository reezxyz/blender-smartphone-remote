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

"""
Status panel, export settings and phone-pose decoding for the smartphone remote.

* A thread-safe state dictionary that network threads update and the panel
  displays (connection, AR frames, scene export / transfer phase).
* Settings stored in the scene: export limits and camera options.
* A panel in the sidebar tab "Remote" (3D Viewport > N panel > Remote).
* Decoding of the phone camera pose into a Blender camera matrix.
"""

import logging
import math
import textwrap
import threading
import time

import bpy
import numpy as np
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty)
from mathutils import Matrix

log = logging.getLogger(__name__)

CONNECTION_TIMEOUT = 8.0     # seconds without ping => phone considered lost
HEAVY_MB = 40                # file size flagged as "HEAVY"
MEDIUM_MB = 10               # file size flagged as "medium"


'''
    Shared state (written by any thread, read by the panel)
'''

_lock = threading.Lock()
_version = 0
_state = dict(
    server="STOPPED",        # STOPPED | RUNNING
    client_addr=None,
    last_ping=0.0,
    frames=0,
    last_frame=0.0,
    fps=0.0,

    phase="IDLE",            # IDLE | EXPORTING | READY | SENDING | DONE | ERROR
    phase_started=0.0,
    message="",

    objects=0,
    tris_before=0,
    tris_after=0,
    decimated=False,
    size_bytes=0,
    export_seconds=0.0,

    sent_bytes=0,
    total_bytes=0,
    send_started=0.0,
    send_seconds=0.0,
)


def update(**kw):
    """Update the shared state (safe to call from any thread)."""
    global _version
    now = time.time()
    with _lock:
        if "phase" in kw and kw["phase"] != _state["phase"]:
            _state["phase_started"] = now
        _state.update(kw)
        _version += 1


def snapshot():
    with _lock:
        return dict(_state)


def count_frame():
    """Called for every AR frame received (network thread)."""
    with _lock:
        _state["frames"] += 1
        _state["last_frame"] = time.time()


def note_ping(client_addr=None):
    """Called for every heartbeat received (network thread)."""
    global _version
    with _lock:
        _state["last_ping"] = time.time()
        if client_addr and _state["client_addr"] != client_addr:
            _state["client_addr"] = client_addr
            _version += 1


def set_server_running(running):
    update(server="RUNNING" if running else "STOPPED",
           client_addr=None, last_ping=0.0, last_frame=0.0, fps=0.0,
           phase="IDLE", message="")


'''
    Settings
'''


class SRExportSettings(bpy.types.PropertyGroup):
    send_model: BoolProperty(
        name="Send model to the phone",
        description="Turn off to use the phone only as a motion tracker "
                    "(gyro / ARCore). No model is exported or sent",
        default=True)
    apply_modifiers: BoolProperty(
        name="Apply modifiers",
        description="Include modifier results (Subdivision, Bevel, ...). "
                    "Turning this off makes the file much smaller",
        default=False)
    max_triangles: IntProperty(
        name="Triangle limit",
        description="Triangle budget for the model sent to the phone "
                    "(0 = no limit)",
        default=150000, min=0, soft_max=2000000)
    limit_action: EnumProperty(
        name="Over the limit",
        items=[
            ('DECIMATE', "Reduce automatically",
             "Use a temporary Decimate to fit the limit "
             "(your scene is not changed)"),
            ('BLOCK', "Do not send",
             "Cancel and show a warning in Blender"),
        ],
        default='DECIMATE')
    max_size_mb: IntProperty(
        name="Max file size (MB)",
        description="Cancel the transfer if the exported file is larger "
                    "(0 = no limit)",
        default=40, min=0)
    export_scale: FloatProperty(
        name="Model scale",
        description="Multiplier for the size of the model sent to the phone",
        default=1.0, min=0.001, soft_max=100.0, precision=3)
    export_mode: EnumProperty(
        name="Objects to export",
        description="Which objects are sent to the phone",
        items=[('VISIBLE', "Visible in viewport",
                "Everything that is visible in the 3D viewport"),
               ('SELECTED', "Selected objects",
                "Only the objects that are currently selected"),
               ('COLLECTION', "A collection",
                "Only the visible objects of one collection"),
               ('RENDERABLE', "Enabled for rendering",
                "Visible objects whose render (camera) icon is enabled")],
        default='VISIBLE')
    export_collection: PointerProperty(
        name="Collection",
        description="Collection to send to the phone",
        type=bpy.types.Collection)


class SRCameraSettings(bpy.types.PropertyGroup):
    position_mode: EnumProperty(
        name="Phone matrix",
        description="What the matrix sent by the phone contains",
        items=[('VIEW', "View matrix (world -> camera)",
                "The phone sends a view matrix"),
               ('POSE', "Camera pose (camera -> world)",
                "The phone sends the camera pose"),
               ('OFF', "Rotation only", "The camera does not change position")],
        default='POSE')
    flip_direction: BoolProperty(
        name="Flip model direction (180°)",
        description="Turn the camera 180° around the vertical axis. "
                    "Toggle this if the model faces the opposite way in "
                    "Blender and on the phone",
        default=True)
    record_max_frames: IntProperty(
        name="Max recorded frames",
        description="Maximum length of a camera recording. The scene end "
                    "frame is raised for the recording and restored after",
        default=3000, min=10, soft_max=20000)
    position_scale: FloatProperty(
        name="Movement scale",
        description="Multiplier for how far the Blender camera moves. Lower "
                    "it if the camera moves further than on the phone",
        default=1.0, min=0.001, soft_max=20.0, precision=3)


class _DefaultSettings:
    send_model = True
    """Used if the scene properties are not available for some reason."""
    apply_modifiers = False
    max_triangles = 150000
    limit_action = 'DECIMATE'
    max_size_mb = 40
    export_scale = 1.0
    export_mode = 'VISIBLE'
    export_collection = None


class _DefaultCamera:
    position_mode = 'POSE'
    flip_direction = True
    position_scale = 1.0
    record_max_frames = 3000


def get_settings():
    try:
        return bpy.context.scene.sr_export
    except Exception:
        return _DefaultSettings


def get_camera_settings():
    try:
        return bpy.context.scene.sr_camera
    except Exception:
        return _DefaultCamera


'''
    Phone pose -> Blender camera matrix

    The phone sends 4x4 matrices as column-major arrays, so numpy sees them
    transposed (the translation is in the last ROW). They are decoded here
    and expressed in the space of the model (the anchor).
'''

# Phone/OBJ world (Y up) -> Blender world (Z up): the inverse of the OBJ
# export (forward -Z, up Y).
W_OBJ = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])

# 180 degrees around the vertical axis of Blender.
YAW_180 = np.diag([-1.0, -1.0, 1.0])


def decode_camera(view, kind):
    """
    Camera orientation (3x3, camera->world) and position in the phone world.
    kind 'VIEW': the matrix is world->camera,  'POSE': it is camera->world.
    """
    n = np.asarray(view, dtype=float)
    if kind == 'POSE':
        return n[:3, :3].T, n[3, :3].copy()
    rot = n[:3, :3]
    return rot, -(rot @ n[3, :3])


def decode_anchor(root):
    """Model (anchor) rotation, position and uniform scale in the phone world."""
    a = np.asarray(root, dtype=float).T
    scale = float(np.linalg.norm(a[:3, 0]))
    if scale < 1e-9 or not np.all(np.isfinite(a)):       # no anchor yet
        return np.eye(3), np.zeros(3), 1.0
    return a[:3, :3] / scale, a[:3, 3].copy(), scale


def camera_pose(view, root):
    """Blender camera matrix for a phone frame."""
    cfg = get_camera_settings()
    kind = 'POSE' if cfg.position_mode == 'POSE' else 'VIEW'

    rot, cam = decode_camera(view, kind)
    ra, anchor, scale = decode_anchor(root)

    # express the camera in the model's (anchor's) space
    rot = ra.T @ rot
    cam = ra.T @ (cam - anchor) / scale
    if cfg.position_mode == 'OFF':
        cam = np.zeros(3)
    cam = cam * cfg.position_scale

    world = (YAW_180 if cfg.flip_direction else np.eye(3)) @ W_OBJ
    rot = world @ rot
    cam = world @ cam

    m = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            m[i][j] = float(rot[i, j])
        m[i][3] = float(cam[i])
    return m


'''
    Scene analysis
'''


def count_triangles(use_modifiers):
    """
    Count the triangles of visible mesh objects.
    Returns (total, [(object, triangles), ...]). Must run in the main thread.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    per_object = []
    total = 0

    for obj in bpy.context.view_layer.objects:
        if obj.type != 'MESH' or not obj.visible_get():
            continue

        mesh = obj.evaluated_get(depsgraph).data if use_modifiers else obj.data
        tris = len(mesh.loops) - 2 * len(mesh.polygons)
        per_object.append((obj, tris))
        total += tris

    return total, per_object


RESOLUTION_PRESETS = {
    'P916': (1080, 1920),
    'P195': (1080, 2340),
    'L169': (1920, 1080),
    'L195': (2340, 1080),
}


class SRResolutionOperator(bpy.types.Operator):
    """Set the render resolution of the Blender camera"""
    bl_idname = "remote.set_resolution"
    bl_label = "Set resolution"

    preset: EnumProperty(items=[('P916', "Portrait 9:16", ""),
                                ('P195', "Portrait 9:19.5", ""),
                                ('L169', "Landscape 16:9", ""),
                                ('L195', "Landscape 19.5:9", "")])

    def execute(self, context):
        width, height = RESOLUTION_PRESETS[self.preset]
        render = context.scene.render
        render.resolution_x = width
        render.resolution_y = height
        render.resolution_percentage = 100
        self.report({'INFO'}, "Resolution set to {} x {}".format(width, height))
        return {'FINISHED'}


class SRSwapResolutionOperator(bpy.types.Operator):
    """Swap the width and the height (portrait <-> landscape)"""
    bl_idname = "remote.swap_resolution"
    bl_label = "Swap portrait / landscape"

    def execute(self, context):
        render = context.scene.render
        render.resolution_x, render.resolution_y = (render.resolution_y,
                                                    render.resolution_x)
        return {'FINISHED'}


'''
    Panel
'''


def _fmt_bytes(n):
    return "{:.1f} MB".format(n / (1024 * 1024)) if n >= 1024 * 1024 \
        else "{:.0f} KB".format(n / 1024)


def _wrap(layout, text, icon='NONE', alert=False, width=38):
    col = layout.column(align=True)
    col.alert = alert
    first = True
    for line in textwrap.wrap(text, width) or [""]:
        col.label(text=line, icon=icon if first else 'BLANK1')
        first = False


def _progress(layout, factor, text):
    try:
        layout.progress(factor=factor, type='BAR', text=text)
    except Exception:
        layout.label(text="{} ({:.0f}%)".format(text, factor * 100))


class SR_PT_status(bpy.types.Panel):
    bl_label = "Status & Settings"
    bl_idname = "REMOTE_PT_sr_status"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Remote"

    def draw(self, context):
        s = snapshot()
        now = time.time()
        self._draw_connection(self.layout, s, now)
        self._draw_scene(self.layout, s, now)
        self._draw_camera(self.layout, context)
        self._draw_export(self.layout)

    # -- connection --------------------------------------------------------
    def _draw_connection(self, layout, s, now):
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Connection", icon='LINKED')

        if s["server"] != "RUNNING":
            col.label(text="Server stopped (press START)", icon='PAUSE')
            return

        col.label(text="Server running", icon='PLAY')
        if not s["last_ping"]:
            col.label(text="Waiting for the phone...", icon='TIME')
        else:
            age = now - s["last_ping"]
            if age > CONNECTION_TIMEOUT:
                row = col.row()
                row.alert = True
                row.label(text="Phone not responding ({:.0f}s)".format(age),
                          icon='ERROR')
            else:
                col.label(text="Phone connected {}".format(
                    s["client_addr"] or ""), icon='CHECKMARK')

        if s["last_frame"] and now - s["last_frame"] < 3:
            col.label(text="Camera frames: {:.0f}/s".format(s["fps"]),
                      icon='CAMERA_DATA')
        else:
            row = col.row()
            row.alert = bool(s["last_ping"]
                             and now - s["last_ping"] <= CONNECTION_TIMEOUT)
            if s["last_frame"]:
                row.label(text="Camera frames stopped {:.0f}s ago".format(
                    now - s["last_frame"]), icon='ERROR')
            else:
                row.label(text="Camera frames: none", icon='CAMERA_DATA')
            if row.alert:
                _wrap(col, "The phone is connected but sends no camera data. "
                           "ARCore probably lost tracking: point the phone "
                           "at a textured surface, or restart the app.",
                      'INFO')

    # -- scene / model -----------------------------------------------------
    def _draw_scene(self, layout, s, now):
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Scene / Model", icon='MESH_DATA')

        phase = s["phase"]
        if not get_settings().send_model:
            col.label(text="Tracking only: no model is sent", icon='INFO')
            return
        if phase == "IDLE":
            col.label(text="No scene request from the phone yet", icon='INFO')
        elif phase == "EXPORTING":
            _wrap(col, s["message"] or "Exporting the scene...", 'EXPORT')
            col.label(text="Running for {:.0f} s".format(
                now - s["phase_started"]), icon='TIME')
        elif phase == "READY":
            col.label(text="Ready to send ({})".format(
                _fmt_bytes(s["size_bytes"])), icon='CHECKMARK')
        elif phase == "SENDING":
            total = max(s["total_bytes"], 1)
            sent = min(s["sent_bytes"], total)
            elapsed = max(now - s["send_started"], 0.001)
            col.label(text="Sending to the phone...", icon='EXPORT')
            _progress(col, sent / total, "{} / {}".format(
                _fmt_bytes(sent), _fmt_bytes(total)))
            col.label(text="{}/s".format(_fmt_bytes(sent / elapsed)))
        elif phase == "DONE":
            col.label(text="Sent in {:.1f} s".format(s["send_seconds"]),
                      icon='CHECKMARK')
            col.label(text="Model not visible? The phone failed to load it",
                      icon='INFO')
        elif phase == "ERROR":
            _wrap(col, s["message"], 'ERROR', alert=True)

        if s["tris_before"]:
            col.separator()
            col.label(text="Objects: {}".format(s["objects"]))
            if s["decimated"]:
                col.label(text="Triangles: {:,} -> {:,} (reduced)".format(
                    s["tris_before"], s["tris_after"]), icon='MOD_DECIM')
            else:
                col.label(text="Triangles: {:,}".format(s["tris_before"]))
        if s["size_bytes"]:
            mb = s["size_bytes"] / (1024 * 1024)
            if mb >= HEAVY_MB:
                row = col.row()
                row.alert = True
                row.label(text="Size {:.0f} MB: HEAVY, phone may freeze"
                          .format(mb), icon='ERROR')
            elif mb >= MEDIUM_MB:
                col.label(text="Size {:.0f} MB: medium".format(mb),
                          icon='INFO')
            else:
                col.label(text="Size {:.1f} MB: light".format(mb),
                          icon='CHECKMARK')
            col.label(text="Export time: {:.1f} s".format(
                s["export_seconds"]))

    # -- camera ------------------------------------------------------------
    def _draw_camera(self, layout, context):
        cfg = get_camera_settings()
        render = context.scene.render
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Camera", icon='CAMERA_DATA')
        col.prop(cfg, "position_mode", text="")
        col.prop(cfg, "flip_direction")
        col.prop(cfg, "position_scale")
        col.prop(cfg, "record_max_frames")

        col.separator()
        col.label(text="Camera resolution")
        row = col.row(align=True)
        row.prop(render, "resolution_x", text="X")
        row.prop(render, "resolution_y", text="Y")
        if render.resolution_y:
            col.label(text="Aspect ratio: {:.3f}".format(
                render.resolution_x / render.resolution_y))
        row = col.row(align=True)
        for preset, text in (('P916', "9:16"), ('P195', "9:19.5"),
                             ('L169', "16:9"), ('L195', "19.5:9")):
            op = row.operator("remote.set_resolution", text=text)
            op.preset = preset
        col.operator("remote.swap_resolution", icon='ARROW_LEFTRIGHT')

    # -- export ------------------------------------------------------------
    def _draw_export(self, layout):
        cfg = get_settings()
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Export to the phone", icon='EXPORT')
        col.prop(cfg, "send_model")
        if not cfg.send_model:
            return
        col.prop(cfg, "export_mode", text="")
        if cfg.export_mode == 'COLLECTION':
            col.prop(cfg, "export_collection", text="")
        col.separator()
        col.prop(cfg, "max_triangles")
        col.prop(cfg, "limit_action", text="")
        col.prop(cfg, "max_size_mb")
        col.prop(cfg, "export_scale")
        col.prop(cfg, "apply_modifiers")


'''
    Periodic redraw (the panel does not refresh by itself)
'''

_last_version = -1
_last_redraw = 0.0
_last_fps_time = 0.0
_last_fps_frames = 0


def _redraw_panels():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


def _ui_refresh():
    global _last_version, _last_redraw, _last_fps_time, _last_fps_frames
    now = time.time()

    with _lock:
        version = _version
        frames = _state["frames"]

    dt = now - _last_fps_time
    if dt >= 1.0:
        with _lock:
            _state["fps"] = (frames - _last_fps_frames) / dt
        _last_fps_time = now
        _last_fps_frames = frames

    if version != _last_version or now - _last_redraw >= 1.0:
        _last_version = version
        _last_redraw = now
        _redraw_panels()

    return 0.2


'''
    Registration
'''

classes = (
    SRExportSettings,
    SRCameraSettings,
    SRResolutionOperator,
    SRSwapResolutionOperator,
    SR_PT_status,
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)
    bpy.types.Scene.sr_export = PointerProperty(type=SRExportSettings)
    bpy.types.Scene.sr_camera = PointerProperty(type=SRCameraSettings)

    if not bpy.app.timers.is_registered(_ui_refresh):
        bpy.app.timers.register(_ui_refresh, first_interval=0.5,
                                persistent=True)


def unregister():
    from bpy.utils import unregister_class

    if bpy.app.timers.is_registered(_ui_refresh):
        bpy.app.timers.unregister(_ui_refresh)

    for name in ("sr_export", "sr_camera"):
        try:
            delattr(bpy.types.Scene, name)
        except Exception:
            pass

    for cls in reversed(classes):
        unregister_class(cls)
