import os
from flask import Flask, render_template, request, send_from_directory
from PIL import Image, ImageOps
import pillow_heif
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
import trimesh.creation as creation

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

            if ext == 'heic':
                heif_file = pillow_heif.read_heif(original_path)
                image = Image.frombytes(
                    heif_file.mode, heif_file.size, heif_file.data, "raw"
                )
            else:
                image = Image.open(original_path).convert('RGB')
                image = ImageOps.exif_transpose(image)

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

    try:
        max_height = float(request.args.get('max_height', '5.0'))
        pixel_scale = float(request.args.get('pixel_scale', '0.1'))
        back_thickness = float(request.args.get('back_thickness', '1.0'))
    except ValueError:
        max_height = 5.0
        pixel_scale = 0.1
        back_thickness = 1.0

    # Flip Y-axis and apply Gaussian smoothing
    pixel_data = np.array(image.transpose(Image.FLIP_TOP_BOTTOM))
    smoothed = gaussian_filter(pixel_data, sigma=1.0)
    # Map brightness to relief thickness: white -> thin, black -> thick
    heightmap = (1.0 - (smoothed / 255.0)) * max_height
    rows, cols = heightmap.shape

    # Define dimensions in millimeters
    width_mm = cols * pixel_scale
    height_mm = rows * pixel_scale

    # Generate grid of vertices for the lithophane surface
    x = np.linspace(0, width_mm, cols)
    y = np.linspace(0, height_mm, rows)
    xx, yy = np.meshgrid(x, y)
    zz = heightmap + back_thickness
    vertices_top = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))

    # Create faces for the top surface mesh
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

    # Create the flat solid backing plate as a box
    base = creation.box(extents=[width_mm, height_mm, back_thickness])
    base.apply_translation([width_mm / 2, height_mm / 2, back_thickness / 2])

    # --- New: create vertical sidewalls to connect the top relief down to the backing plate ---
    # Duplicate top-surface vertices at z = back_thickness for bottom ring
    num_top = vertices_top.shape[0]
    vertices_bottom = vertices_top.copy()
    vertices_bottom[:, 2] = back_thickness

    sw_vertices = np.vstack((vertices_top, vertices_bottom))
    bottom_offset = num_top

    side_faces = []
    # Top edge (y = 0)
    for j in range(cols - 1):
        v0 = j
        v1 = j + 1
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.append([v0, v1, v2])
        side_faces.append([v0, v2, v3])
    # Bottom edge (y = max)
    for j in range(cols - 1):
        idx = (rows - 1) * cols + j
        v0 = idx
        v1 = idx + 1
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.append([v1, v0, v2])
        side_faces.append([v0, v3, v2])
    # Left edge (x = 0)
    for i in range(rows - 1):
        idx = i * cols
        v0 = idx
        v1 = idx + cols
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.append([v1, v0, v2])
        side_faces.append([v0, v3, v2])
    # Right edge (x = max)
    for i in range(rows - 1):
        idx = i * cols + (cols - 1)
        v0 = idx
        v1 = idx + cols
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.append([v0, v1, v2])
        side_faces.append([v0, v2, v3])

    side_mesh = trimesh.Trimesh(vertices=sw_vertices, faces=side_faces, process=True)

    # Combine backing plate, top relief surface, and sidewalls into one watertight solid
    combined = trimesh.util.concatenate([base, top_surface, side_mesh])

    # Export as STL
    stl_filename = filename.replace('.png', '.stl')
    stl_path = os.path.join(app.config['UPLOAD_FOLDER'], stl_filename)
    combined.export(stl_path)

    return send_from_directory(app.config['UPLOAD_FOLDER'], stl_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
