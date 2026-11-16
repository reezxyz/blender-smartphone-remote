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
Status panel + export settings for the smartphone remote.

* A thread-safe state dictionary that network threads update and the
  panel displays (connection, AR frames, scene export / transfer phase).
* Export settings stored in the scene (triangle budget, size cap, ...).
* A panel in the sidebar tab "Remote" (View3D > N panel > Remote).
"""

import logging
import textwrap
import threading
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty

log = logging.getLogger(__name__)

CONNECTION_TIMEOUT = 8.0     # seconds without ping => phone considered lost
HEAVY_MB = 40                # file size flagged as "BERAT"
MEDIUM_MB = 10               # file size flagged as "Sedang"


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

    an_time=0.0,             # last manual analysis
    an_objects=0,
    an_tris=0,
    an_modifiers=False,
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
    Export settings
'''


class SRExportSettings(bpy.types.PropertyGroup):
    apply_modifiers: BoolProperty(
        name="Terapkan modifier",
        description="Sertakan hasil modifier (Subdivision, Bevel, ...). "
                    "Mematikan ini membuat file jauh lebih kecil",
        default=False)
    export_normals: BoolProperty(
        name="Normal", description="Ekspor normal", default=True)
    export_uv: BoolProperty(
        name="UV", description="Ekspor koordinat UV", default=True)
    triangulate: BoolProperty(
        name="Triangulasi", description="Ubah semua face menjadi segitiga",
        default=True)
    selected_only: BoolProperty(
        name="Hanya objek terpilih",
        description="Hanya kirim objek yang sedang dipilih", default=False)
    max_triangles: IntProperty(
        name="Batas triangle",
        description="Anggaran triangle untuk dikirim ke HP (0 = tanpa batas)",
        default=150000, min=0, soft_max=2000000)
    limit_action: EnumProperty(
        name="Jika melebihi batas",
        items=[
            ('DECIMATE', "Kecilkan otomatis",
             "Pakai Decimate sementara agar muat dalam batas "
             "(scene asli tidak diubah)"),
            ('BLOCK', "Jangan kirim",
             "Batalkan dan tampilkan peringatan di Blender"),
        ],
        default='DECIMATE')
    max_size_mb: IntProperty(
        name="Batas ukuran file (MB)",
        description="Batalkan pengiriman jika hasil ekspor lebih besar "
                    "(0 = tanpa batas)",
        default=40, min=0)


class _DefaultSettings:
    """Used if the scene property is not available for some reason."""
    apply_modifiers = False
    export_normals = True
    export_uv = True
    triangulate = True
    selected_only = False
    max_triangles = 150000
    limit_action = 'DECIMATE'
    max_size_mb = 40


def get_settings():
    try:
        return bpy.context.scene.sr_export
    except Exception:
        return _DefaultSettings


'''
    Scene analysis
'''


def count_triangles(use_modifiers, selected_only):
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
        if selected_only and not obj.select_get():
            continue

        mesh = obj.evaluated_get(depsgraph).data if use_modifiers else obj.data
        tris = len(mesh.loops) - 2 * len(mesh.polygons)
        per_object.append((obj, tris))
        total += tris

    return total, per_object


class SRAnalyzeOperator(bpy.types.Operator):
    """Hitung jumlah triangle scene dengan pengaturan ekspor saat ini"""
    bl_idname = "remote.analyze_scene"
    bl_label = "Analisis scene"

    def execute(self, context):
        cfg = get_settings()
        tris, per_object = count_triangles(cfg.apply_modifiers,
                                           cfg.selected_only)
        update(an_time=time.time(), an_objects=len(per_object),
               an_tris=tris, an_modifiers=cfg.apply_modifiers)
        self.report({'INFO'}, "{:,} triangle di {} objek".format(
            tris, len(per_object)))
        return {'FINISHED'}


PRESETS = {
    'LIGHT': dict(apply_modifiers=False, max_triangles=100000,
                  limit_action='DECIMATE', max_size_mb=20),
    'MEDIUM': dict(apply_modifiers=False, max_triangles=300000,
                   limit_action='DECIMATE', max_size_mb=50),
    'FULL': dict(apply_modifiers=True, max_triangles=0,
                 limit_action='DECIMATE', max_size_mb=0),
}


class SRPresetOperator(bpy.types.Operator):
    """Terapkan preset kualitas ekspor"""
    bl_idname = "remote.apply_export_preset"
    bl_label = "Preset ekspor"

    preset: EnumProperty(items=[('LIGHT', "Ringan", ""),
                                ('MEDIUM', "Sedang", ""),
                                ('FULL', "Penuh", "")])

    def execute(self, context):
        cfg = context.scene.sr_export
        for key, value in PRESETS[self.preset].items():
            setattr(cfg, key, value)
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
    bl_label = "Status & Ekspor"
    bl_idname = "REMOTE_PT_sr_status"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Remote"

    def draw(self, context):
        s = snapshot()
        now = time.time()
        self._draw_connection(self.layout, s, now)
        self._draw_scene(self.layout, s, now)
        self._draw_options(self.layout, context, s)

    # -- connection --------------------------------------------------------
    def _draw_connection(self, layout, s, now):
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Koneksi", icon='LINKED')

        if s["server"] != "RUNNING":
            col.label(text="Server berhenti (tekan START)", icon='PAUSE')
            return

        col.label(text="Server berjalan", icon='PLAY')
        if not s["last_ping"]:
            col.label(text="Menunggu HP terhubung...", icon='TIME')
        else:
            age = now - s["last_ping"]
            if age > CONNECTION_TIMEOUT:
                row = col.row()
                row.alert = True
                row.label(text="HP tidak merespons ({:.0f}s)".format(age),
                          icon='ERROR')
            else:
                col.label(text="HP terhubung {}".format(
                    s["client_addr"] or ""), icon='CHECKMARK')

        if s["last_frame"] and now - s["last_frame"] < 3:
            col.label(text="Frame kamera: {:.0f}/detik".format(s["fps"]),
                      icon='CAMERA_DATA')
        else:
            col.label(text="Frame kamera: tidak ada", icon='CAMERA_DATA')

    # -- scene / model -----------------------------------------------------
    def _draw_scene(self, layout, s, now):
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Scene / Model", icon='MESH_DATA')

        phase = s["phase"]
        if phase == "IDLE":
            col.label(text="Belum ada permintaan dari HP", icon='INFO')
        elif phase == "EXPORTING":
            _wrap(col, s["message"] or "Mengekspor scene...", 'EXPORT')
            col.label(text="Sudah {:.0f} detik".format(
                now - s["phase_started"]), icon='TIME')
        elif phase == "READY":
            col.label(text="Siap dikirim ({})".format(
                _fmt_bytes(s["size_bytes"])), icon='CHECKMARK')
        elif phase == "SENDING":
            total = max(s["total_bytes"], 1)
            sent = min(s["sent_bytes"], total)
            elapsed = max(now - s["send_started"], 0.001)
            speed = sent / elapsed
            col.label(text="Mengirim ke HP...", icon='EXPORT')
            _progress(col, sent / total, "{} / {}".format(
                _fmt_bytes(sent), _fmt_bytes(total)))
            col.label(text="{}/detik".format(_fmt_bytes(speed)))
        elif phase == "DONE":
            col.label(text="Selesai dikirim ({:.1f}s)".format(
                s["send_seconds"]), icon='CHECKMARK')
            col.label(text="Jika model belum muncul, HP yang memprosesnya",
                      icon='INFO')
        elif phase == "ERROR":
            _wrap(col, s["message"], 'ERROR', alert=True)

        if s["tris_before"]:
            col.separator()
            col.label(text="Objek: {}".format(s["objects"]))
            if s["decimated"]:
                col.label(text="Triangle: {:,} -> {:,} (dikecilkan)".format(
                    s["tris_before"], s["tris_after"]), icon='MOD_DECIM')
            else:
                col.label(text="Triangle: {:,}".format(s["tris_before"]))
        if s["size_bytes"]:
            mb = s["size_bytes"] / (1024 * 1024)
            if mb >= HEAVY_MB:
                row = col.row()
                row.alert = True
                row.label(text="Ukuran {:.0f} MB: BERAT, HP bisa hang".format(
                    mb), icon='ERROR')
            elif mb >= MEDIUM_MB:
                col.label(text="Ukuran {:.0f} MB: sedang".format(mb),
                          icon='INFO')
            else:
                col.label(text="Ukuran {:.1f} MB: ringan".format(mb),
                          icon='CHECKMARK')
            col.label(text="Waktu ekspor: {:.1f}s".format(
                s["export_seconds"]))

    # -- options -----------------------------------------------------------
    def _draw_options(self, layout, context, s):
        cfg = get_settings()
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Opsi ekspor ke HP", icon='PREFERENCES')

        row = col.row(align=True)
        for key, text in (('LIGHT', "Ringan"), ('MEDIUM', "Sedang"),
                          ('FULL', "Penuh")):
            op = row.operator("remote.apply_export_preset", text=text)
            op.preset = key

        col.separator()
        col.prop(cfg, "max_triangles")
        col.prop(cfg, "limit_action", text="")
        col.prop(cfg, "max_size_mb")
        col.separator()
        col.prop(cfg, "apply_modifiers")
        col.prop(cfg, "selected_only")
        row = col.row(align=True)
        row.prop(cfg, "export_normals", toggle=True)
        row.prop(cfg, "export_uv", toggle=True)
        row.prop(cfg, "triangulate", toggle=True)

        col.separator()
        col.operator("remote.analyze_scene", icon='VIEWZOOM')
        if s["an_time"]:
            col.label(text="Hasil: {:,} triangle, {} objek".format(
                s["an_tris"], s["an_objects"]))
            limit = cfg.max_triangles
            if not limit:
                col.label(text="Tanpa batas: model besar bisa membuat HP hang",
                          icon='INFO')
            elif s["an_tris"] <= limit:
                col.label(text="Di bawah batas: aman dikirim",
                          icon='CHECKMARK')
            elif cfg.limit_action == 'DECIMATE':
                _wrap(col, "Melebihi batas: akan dikecilkan otomatis ke "
                           "sekitar {:,} triangle".format(limit), 'MOD_DECIM')
            else:
                _wrap(col, "Melebihi batas: scene TIDAK akan dikirim",
                      'ERROR', alert=True)


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
    SRAnalyzeOperator,
    SRPresetOperator,
    SR_PT_status,
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)
    bpy.types.Scene.sr_export = PointerProperty(type=SRExportSettings)

    if not bpy.app.timers.is_registered(_ui_refresh):
        bpy.app.timers.register(_ui_refresh, first_interval=0.5,
                                persistent=True)


def unregister():
    from bpy.utils import unregister_class

    if bpy.app.timers.is_registered(_ui_refresh):
        bpy.app.timers.unregister(_ui_refresh)

    try:
        del bpy.types.Scene.sr_export
    except Exception:
        pass

    for cls in reversed(classes):
        unregister_class(cls)
