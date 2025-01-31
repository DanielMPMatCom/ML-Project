import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os

# Configuración inicial
input_folder = r"E:\Universidad\ML\ML-Project\data\Generales-Parciales"
output_folder = os.path.join(input_folder, "annotations")
os.makedirs(output_folder, exist_ok=True)

# Variables globales
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
current_index = 0
rect_start = None
rect_end = None
rect_id = None
zoom_level = 1.0
current_image = None
img_tk = None
offset_x = 0
offset_y = 0
pan_start = None

# Funciones principales
def load_image(index):
    if 0 <= index < len(image_files):
        image_path = os.path.join(input_folder, image_files[index])
        image = Image.open(image_path)
        return image
    return None

def save_rectangle(image, start, end, output_path):
    cropped = image.crop((min(start[0], end[0]), min(start[1], end[1]),
                          max(start[0], end[0]), max(start[1], end[1])))
    cropped.save(output_path)

def on_mouse_press(event):
    global rect_start, rect_id, pan_start
    if event.num == 1:  # Left click
        rect_start = (int((event.x - offset_x) / zoom_level), int((event.y - offset_y) / zoom_level))
        rect_id = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red")
    elif event.num == 3:  # Right click
        pan_start = (event.x, event.y)

def on_mouse_drag(event):
    global rect_id, offset_x, offset_y, pan_start
    if rect_id and rect_start:
        canvas.coords(rect_id, rect_start[0] * zoom_level + offset_x, rect_start[1] * zoom_level + offset_y, event.x, event.y)
    elif pan_start:
        dx = event.x - pan_start[0]
        dy = event.y - pan_start[1]
        offset_x += dx
        offset_y += dy
        pan_start = (event.x, event.y)
        update_canvas()

def on_mouse_release(event):
    global rect_start, rect_end, rect_id, current_image, pan_start
    if event.num == 1 and rect_start:  # Left click release
        rect_end = (int((event.x - offset_x) / zoom_level), int((event.y - offset_y) / zoom_level))
        if rect_start and rect_end:
            output_path = os.path.join(output_folder, f"annotation_{current_index}_{rect_start[0]}_{rect_start[1]}_{rect_end[0]}_{rect_end[1]}.png")
            save_rectangle(current_image, rect_start, rect_end, output_path)
        rect_start = None
        rect_end = None
        rect_id = None
    elif event.num == 3:  # Right click release
        pan_start = None

def on_mouse_wheel(event):
    global zoom_level, offset_x, offset_y
    factor = 1.1 if event.delta > 0 else 0.9
    new_zoom_level = zoom_level * factor

    # Adjust offsets to zoom relative to the cursor position
    cursor_x = canvas.canvasx(event.x)
    cursor_y = canvas.canvasy(event.y)
    offset_x = cursor_x - factor * (cursor_x - offset_x)
    offset_y = cursor_y - factor * (cursor_y - offset_y)

    zoom_level = new_zoom_level
    update_canvas()

def next_image():
    global current_index, current_image, img_tk, zoom_level, offset_x, offset_y
    current_index += 1
    zoom_level = 1.0
    offset_x = 0
    offset_y = 0
    if current_index < len(image_files):
        current_image = load_image(current_index)
        update_canvas()
    else:
        print("No hay más imágenes.")

def update_canvas():
    global img_tk
    if current_image:
        resized_image = current_image.resize((int(current_image.width * zoom_level), int(current_image.height * zoom_level)))
        img_tk = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.config(scrollregion=(0, 0, resized_image.width + offset_x, resized_image.height + offset_y))
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=img_tk)

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Herramienta de Anotación Rápida")

canvas = tk.Canvas(root)
canvas.pack(fill=tk.BOTH, expand=True)

frame_buttons = tk.Frame(root)
frame_buttons.pack()

btn_next = tk.Button(frame_buttons, text="Siguiente Imagen", command=next_image)
btn_next.pack(side=tk.LEFT)

canvas.bind("<ButtonPress-1>", on_mouse_press)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_release)
canvas.bind("<MouseWheel>", on_mouse_wheel)
canvas.bind("<ButtonPress-3>", on_mouse_press)
canvas.bind("<B3-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-3>", on_mouse_release)

# Cargar la primera imagen
if image_files:
    current_image = load_image(current_index)
    update_canvas()
else:
    print("No se encontraron imágenes en la carpeta especificada.")

root.mainloop()
