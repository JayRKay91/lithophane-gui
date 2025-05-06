import os
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
from flask import Flask, render_template, request, send_from_directory
from PIL import Image, ImageOps, ImageTk
import pillow_heif
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
import trimesh.creation as creation

# ─── Flask App Configuration ────────────────────────────────────────────────────

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    grayscale_filename = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(original_path)

            # Handle HEIC separately
            if ext == 'heic':
                heif_file = pillow_heif.read_heif(original_path)
                image = Image.frombytes(
                    heif_file.mode, heif_file.size, heif_file.data, "raw"
                )
            else:
                image = Image.open(original_path).convert('RGB')
                image = ImageOps.exif_transpose(image)

            # Convert to grayscale and save
            grayscale = image.convert('L')
            grayscale_filename = f"grayscale_{file.filename.rsplit('.', 1)[0]}.png"
            grayscale_path = os.path.join(app.config['UPLOAD_FOLDER'], grayscale_filename)
            grayscale.save(grayscale_path)

            params = {
                'max_height': request.form.get('max_height', '5.0'),
                'pixel_scale': request.form.get('pixel_scale', '0.1'),
                'back_thickness': request.form.get('back_thickness', '1.0')
            }

            return render_template('index.html', grayscale_image=grayscale_filename, params=params)

    return render_template('index.html', grayscale_image=None, params=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/generate_stl/<filename>')
def generate_stl(filename):
    grayscale_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(grayscale_path).convert('L')
    image = ImageOps.exif_transpose(image)

    # Parse parameters
    try:
        max_height = float(request.args.get('max_height', '5.0'))
        pixel_scale = float(request.args.get('pixel_scale', '0.1'))
        back_thickness = float(request.args.get('back_thickness', '1.0'))
    except ValueError:
        max_height = 5.0
        pixel_scale = 0.1
        back_thickness = 1.0

    # Prepare heightmap
    pixel_data = np.array(image.transpose(Image.FLIP_TOP_BOTTOM))
    smoothed = gaussian_filter(pixel_data, sigma=1.0)
    heightmap = (1.0 - (smoothed / 255.0)) * max_height
    rows, cols = heightmap.shape

    # Dimensions in mm
    width_mm = cols * pixel_scale
    height_mm = rows * pixel_scale

    # Create top surface vertices
    x = np.linspace(0, width_mm, cols)
    y = np.linspace(0, height_mm, rows)
    xx, yy = np.meshgrid(x, y)
    zz = heightmap + back_thickness
    vertices_top = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))

    # Create faces for top surface
    faces_top = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            v0 = idx
            v1 = idx + 1
            v2 = idx + cols
            v3 = idx + cols + 1
            faces_top.append([v0, v2, v1])
            faces_top.append([v1, v2, v3])
    top_surface = trimesh.Trimesh(vertices=vertices_top, faces=faces_top, process=True)

    # Create backing plate
    base = creation.box(extents=[width_mm, height_mm, back_thickness])
    base.apply_translation([width_mm / 2, height_mm / 2, back_thickness / 2])

    # Build sidewalls
    num_top = vertices_top.shape[0]
    vertices_bottom = vertices_top.copy()
    vertices_bottom[:, 2] = back_thickness
    sw_vertices = np.vstack((vertices_top, vertices_bottom))
    bottom_offset = num_top

    side_faces = []
    # Top edge
    for j in range(cols - 1):
        v0 = j
        v1 = j + 1
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.extend([[v0, v1, v2], [v0, v2, v3]])
    # Bottom edge
    for j in range(cols - 1):
        idx = (rows - 1) * cols + j
        v0 = idx
        v1 = idx + 1
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.extend([[v1, v0, v2], [v0, v3, v2]])
    # Left edge
    for i in range(rows - 1):
        idx = i * cols
        v0 = idx
        v1 = idx + cols
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.extend([[v1, v0, v2], [v0, v3, v2]])
    # Right edge
    for i in range(rows - 1):
        idx = i * cols + (cols - 1)
        v0 = idx
        v1 = idx + cols
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.extend([[v0, v1, v2], [v0, v2, v3]])

    side_mesh = trimesh.Trimesh(vertices=sw_vertices, faces=side_faces, process=True)

    # Combine everything
    combined = trimesh.util.concatenate([base, top_surface, side_mesh])

    # Export STL
    stl_filename = filename.replace('.png', '.stl')
    stl_path = os.path.join(app.config['UPLOAD_FOLDER'], stl_filename)
    combined.export(stl_path)

    return send_from_directory(app.config['UPLOAD_FOLDER'], stl_filename, as_attachment=True)

# ─── Tkinter GUI ────────────────────────────────────────────────────────────────

class LithophaneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lithophane Generator v1.0")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        self.selected_image_path = None
        self.original_image = None

        self.setup_ui()

    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="Lithophane Generator", font=("Helvetica", 16, "bold"))
        title.pack(pady=10)

        # Image preview area
        self.image_label = tk.Label(
            self.root,
            text="No image loaded",
            bg="#e0e0e0",
            width=60,
            height=20
        )
        self.image_label.pack(pady=10)

        # Button to select image
        select_btn = tk.Button(self.root, text="Select Image", command=self.load_image)
        select_btn.pack(pady=5)

        # Button to generate lithophane (placeholder)
        generate_btn = tk.Button(self.root, text="Generate Lithophane", command=self.generate_lithophane)
        generate_btn.pack(pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.heic")]
        )
        if file_path:
            self.selected_image_path = file_path
            try:
                image = Image.open(file_path)
                image.thumbnail((400, 400))  # Resize for display
                self.original_image = image

                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Prevent garbage collection
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def generate_lithophane(self):
        if not self.selected_image_path:
            messagebox.showwarning("No image", "Please select an image first.")
            return
        # Placeholder for actual lithophane logic
        messagebox.showinfo(
            "Processing",
            f"Generating lithophane from:\n{os.path.basename(self.selected_image_path)}"
        )

# ─── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Lithophane app as Flask server or Tkinter GUI'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch the Tkinter GUI instead of the Flask server'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Flask host (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Flask port (default: 5000)'
    )
    args = parser.parse_args()

    if args.gui:
        root = tk.Tk()
        gui_app = LithophaneApp(root)
        root.mainloop()
    else:
        app.run(debug=True, host=args.host, port=args.port)
