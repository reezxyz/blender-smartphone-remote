import bpy
import bgl
import gpu
from gpu_extras.presets import draw_texture_2d
import ffmpeg
from PIL import Image

IMAGE_NAME = "Generated Image"

WIDTH = 512
HEIGHT = 256

offscreen = None
handler = None
buffer = bgl.Buffer(bgl.GL_BYTE, WIDTH * HEIGHT * 3)

def buffer_print():
    global buffer
    # img = Image.frombytes('RGBA', (WIDTH, HEIGHT),buffer)
    # img.save('test.jpg','JPEG')
    # print(buffer)
    return 20

def draw():
    global offscreen, buffer

    context = bpy.context
    scene = context.scene

    view_matrix = scene.camera.matrix_world.inverted()

    projection_matrix = scene.camera.calc_matrix_camera(
        context.evaluated_depsgraph_get(), x=WIDTH, y=HEIGHT)

    offscreen.draw_view3d(
        scene,
        context.view_layer,
        context.space_data,
        context.region,
        view_matrix,
        projection_matrix)

    bgl.glDisable(bgl.GL_DEPTH_TEST)
    draw_texture_2d(offscreen.color_texture, (10, 10), WIDTH, HEIGHT)
    with offscreen.bind():
        bgl.glReadBuffer(bgl.GL_BACK)
        bgl.glReadPixels(0, 0, WIDTH, HEIGHT, bgl.GL_RGB, bgl.GL_UNSIGNED_BYTE, buffer)
        
        # if not IMAGE_NAME in bpy.data.images:
        #     bpy.data.images.new(IMAGE_NAME, WIDTH, HEIGHT)
        # image = bpy.data.images[IMAGE_NAME]
        # image.scale(WIDTH, HEIGHT)
        # image.pixels = [v / 255 for v in buffer]
    # bgl.glGetTexImage(offscreen.color_texture,0, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, buffer)
    

def enable():
    global handler, offscreen

    offscreen = gpu.types.GPUOffScreen(WIDTH, HEIGHT)
    
    handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_PIXEL')
    # bpy.app.timers.register(buffer_print)
    

def disable():
    global handler,offscreen

    if handler:
        bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')

    # bpy.app.timers.unregister(buffer_print)
    offscreen.free()