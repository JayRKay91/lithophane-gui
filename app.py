import os
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from flask import Flask, render_template, request, send_from_directory
from PIL import Image, ImageOps, ImageTk
import pillow_heif
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
import trimesh.creation as creation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

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
                'back_thickness': request.form.get('back_thickness', '1.0'),
                'plate_shape': request.form.get('plate_shape', 'Rectangle'),
                'length': request.form.get('length', '100'),
                'width': request.form.get('width', '100'),
                'diameter': request.form.get('diameter', '100')
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
        plate_shape = request.args.get('plate_shape', 'Rectangle')
        
        if plate_shape == 'Rectangle':
            length = float(request.args.get('length', '100'))
            width = float(request.args.get('width', '100'))
            generate_rectangular_stl_from_image(image, max_height, pixel_scale, back_thickness, length, width, 
                                             os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('.png', '.stl')))
        else:  # Circle
            diameter = float(request.args.get('diameter', '100'))
            generate_circular_stl_from_image(image, max_height, pixel_scale, back_thickness, diameter, 
                                          os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('.png', '.stl')))
    except ValueError as e:
        print(f"Value error: {e}")
        max_height = 5.0
        pixel_scale = 0.1
        back_thickness = 1.0
        generate_rectangular_stl_from_image(image, max_height, pixel_scale, back_thickness, 100, 100,
                                         os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('.png', '.stl')))

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename.replace('.png', '.stl'), as_attachment=True)

# ─── STL Generator Functions ────────────────────────────────────────────────────

def generate_rectangular_stl_from_image(image, max_height, pixel_scale, back_thickness, length, width, output_path):
    # Resize image to fit the dimensions
    target_width = int(width / pixel_scale)
    target_height = int(length / pixel_scale)
    image = image.resize((target_width, target_height), Image.LANCZOS)
    
    pixel_data = np.array(image.transpose(Image.FLIP_TOP_BOTTOM))
    smoothed = gaussian_filter(pixel_data, sigma=1.0)
    heightmap = (1.0 - (smoothed / 255.0)) * max_height
    rows, cols = heightmap.shape

    width_mm = cols * pixel_scale
    height_mm = rows * pixel_scale

    x = np.linspace(0, width_mm, cols)
    y = np.linspace(0, height_mm, rows)
    xx, yy = np.meshgrid(x, y)
    zz = heightmap + back_thickness
    vertices_top = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))

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

    base = creation.box(extents=[width_mm, height_mm, back_thickness])
    base.apply_translation([width_mm / 2, height_mm / 2, back_thickness / 2])

    num_top = vertices_top.shape[0]
    vertices_bottom = vertices_top.copy()
    vertices_bottom[:, 2] = back_thickness
    sw_vertices = np.vstack((vertices_top, vertices_bottom))
    bottom_offset = num_top

    side_faces = []
    for j in range(cols - 1):
        v0 = j
        v1 = j + 1
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.extend([[v0, v1, v2], [v0, v2, v3]])
    for j in range(cols - 1):
        idx = (rows - 1) * cols + j
        v0 = idx
        v1 = idx + 1
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.extend([[v1, v0, v2], [v0, v3, v2]])
    for i in range(rows - 1):
        idx = i * cols
        v0 = idx
        v1 = idx + cols
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.extend([[v1, v0, v2], [v0, v3, v2]])
    for i in range(rows - 1):
        idx = i * cols + (cols - 1)
        v0 = idx
        v1 = idx + cols
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.extend([[v0, v1, v2], [v0, v2, v3]])

    side_mesh = trimesh.Trimesh(vertices=sw_vertices, faces=side_faces, process=True)
    combined = trimesh.util.concatenate([base, top_surface, side_mesh])
    combined.export(output_path)

def generate_circular_stl_from_image(image, max_height, pixel_scale, back_thickness, diameter, output_path):
    # Resize and crop image to fit within a circle
    radius = diameter / 2
    size = int(diameter / pixel_scale)
    
    # Resize the image maintaining aspect ratio
    img_width, img_height = image.size
    aspect_ratio = img_width / img_height
    
    if aspect_ratio > 1:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_height = size
        new_width = int(size * aspect_ratio)
    
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a square image with the circle in the center
    square_img = Image.new('L', (size, size), 255)
    offset = ((size - new_width) // 2, (size - new_height) // 2)
    square_img.paste(image, offset)
    
    # Create a circular mask
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageOps.invert(mask)
    center = size // 2
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            if dist <= center:
                mask_draw.putpixel((x, y), 255)
    
    # Apply mask to image
    circular_img = Image.new('L', (size, size), 255)
    circular_img.paste(square_img, (0, 0), mask)
    
    # Generate the 3D model
    pixel_data = np.array(circular_img.transpose(Image.FLIP_TOP_BOTTOM))
    smoothed = gaussian_filter(pixel_data, sigma=1.0)
    heightmap = (1.0 - (smoothed / 255.0)) * max_height
    
    rows, cols = heightmap.shape
    x = np.linspace(-radius, radius, cols)
    y = np.linspace(-radius, radius, rows)
    xx, yy = np.meshgrid(x, y)
    zz = heightmap + back_thickness
    vertices_top = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    
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
    
    # Create a mask for vertices inside the circle
    center_x = cols // 2
    center_y = rows // 2
    distance_from_center = np.sqrt((xx - 0)**2 + (yy - 0)**2)
    mask = distance_from_center <= radius
    mask_flat = mask.flatten()
    
    # Filter vertices and adjust faces
    valid_indices = np.where(mask_flat)[0]
    index_map = np.full(vertices_top.shape[0], -1)
    index_map[valid_indices] = np.arange(len(valid_indices))
    
    vertices_top_filtered = vertices_top[valid_indices]
    
    # Filter faces to only include those with all vertices inside the circle
    valid_faces = []
    for face in faces_top:
        if all(mask_flat[v] for v in face):
            mapped_face = [index_map[v] for v in face]
            valid_faces.append(mapped_face)
    
    top_surface = trimesh.Trimesh(vertices=vertices_top_filtered, faces=valid_faces, process=True)
    
    # Create circular base
    base = creation.cylinder(radius=radius, height=back_thickness, sections=64)
    base.apply_translation([0, 0, back_thickness / 2])
    
    # Create the sides (connecting top surface to base)
    num_top = len(valid_indices)
    vertices_bottom = vertices_top_filtered.copy()
    vertices_bottom[:, 2] = back_thickness
    sw_vertices = np.vstack((vertices_top_filtered, vertices_bottom))
    bottom_offset = num_top
    
    # We need to create faces that connect the perimeter of the top surface
    # to the perimeter of the base
    perimeter_indices = []
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if not mask_flat[idx]:
                continue
                
            # Check if this is a boundary vertex
            is_boundary = False
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_idx = ni * cols + nj
                    if not mask_flat[neighbor_idx]:
                        is_boundary = True
                        break
                else:
                    is_boundary = True
            
            if is_boundary:
                perimeter_indices.append(index_map[idx])
    
    # Sort perimeter indices to form a continuous loop
    if perimeter_indices:
        # This is a simplified approach - a more robust method would be needed
        # for complex shapes, but for a circle this works reasonably well
        center = np.mean(vertices_top_filtered[perimeter_indices, :2], axis=0)
        angles = np.arctan2(
            vertices_top_filtered[perimeter_indices, 1] - center[1],
            vertices_top_filtered[perimeter_indices, 0] - center[0]
        )
        sorted_indices = [x for _, x in sorted(zip(angles, perimeter_indices))]
        perimeter_indices = sorted_indices
    
    side_faces = []
    num_perimeter = len(perimeter_indices)
    for i in range(num_perimeter):
        v0 = perimeter_indices[i]
        v1 = perimeter_indices[(i + 1) % num_perimeter]
        v2 = bottom_offset + v1
        v3 = bottom_offset + v0
        side_faces.append([v0, v1, v2])
        side_faces.append([v0, v2, v3])
    
    side_mesh = trimesh.Trimesh(vertices=sw_vertices, faces=side_faces, process=True)
    
    # Combine all meshes
    combined = trimesh.util.concatenate([base, top_surface, side_mesh])
    combined.export(output_path)

# ─── Tkinter GUI ────────────────────────────────────────────────────────────────

class LithophaneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lithophane Generator v1.1")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)

        self.selected_image_path = None
        self.original_image = None
        self.canvas = None
        self.plate_shape = tk.StringVar(value="Rectangle")

        self.setup_ui()

    def setup_ui(self):
        title = tk.Label(self.root, text="Lithophane Generator", font=("Helvetica", 16, "bold"))
        title.pack(pady=10)

        frame = tk.Frame(self.root)
        frame.pack(pady=5)

        self.image_label = tk.Label(frame, text="No image loaded", bg="#e0e0e0", width=40, height=15)
        self.image_label.grid(row=0, column=0, padx=10)

        self.preview_frame = tk.Frame(frame, width=400, height=300, bg="#d0d0d0")
        self.preview_frame.grid(row=0, column=1, padx=10)

        select_btn = tk.Button(self.root, text="Upload Image", command=self.load_image)
        select_btn.pack(pady=5)

        settings_frame = tk.LabelFrame(self.root, text="Settings", padx=10, pady=10)
        settings_frame.pack(pady=10)

        # Shape selection
        tk.Label(settings_frame, text="Plate Shape:").grid(row=0, column=0, sticky="e")
        self.shape_cb = ttk.Combobox(settings_frame, textvariable=self.plate_shape, values=["Rectangle", "Circle"], width=10)
        self.shape_cb.grid(row=0, column=1, padx=5)
        self.shape_cb.bind("<<ComboboxSelected>>", self.on_shape_changed)

        # Common settings
        tk.Label(settings_frame, text="Max Height (mm):").grid(row=1, column=0, sticky="e")
        self.max_height_entry = tk.Entry(settings_frame, width=10)
        self.max_height_entry.insert(0, "5.0")
        self.max_height_entry.grid(row=1, column=1, padx=5)

        tk.Label(settings_frame, text="Pixel Scale (mm/pixel):").grid(row=2, column=0, sticky="e")
        self.pixel_scale_entry = tk.Entry(settings_frame, width=10)
        self.pixel_scale_entry.insert(0, "0.1")
        self.pixel_scale_entry.grid(row=2, column=1, padx=5)

        tk.Label(settings_frame, text="Thickness (mm):").grid(row=3, column=0, sticky="e")
        self.thickness_cb = ttk.Combobox(settings_frame, values=["1.0", "2.0", "3.0"], width=7)
        self.thickness_cb.set("1.0")
        self.thickness_cb.grid(row=3, column=1, padx=5)

        # Rectangle specific settings
        self.rectangle_frame = tk.Frame(settings_frame)
        self.rectangle_frame.grid(row=4, column=0, columnspan=2, pady=5)
        
        tk.Label(self.rectangle_frame, text="Length (mm):").grid(row=0, column=0, sticky="e")
        self.length_entry = tk.Entry(self.rectangle_frame, width=10)
        self.length_entry.insert(0, "100")
        self.length_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(self.rectangle_frame, text="Width (mm):").grid(row=1, column=0, sticky="e")
        self.width_entry = tk.Entry(self.rectangle_frame, width=10)
        self.width_entry.insert(0, "100")
        self.width_entry.grid(row=1, column=1, padx=5)
        
        # Circle specific settings
        self.circle_frame = tk.Frame(settings_frame)
        self.circle_frame.grid(row=5, column=0, columnspan=2, pady=5)
        self.circle_frame.grid_remove()  # Hide initially
        
        tk.Label(self.circle_frame, text="Diameter (mm):").grid(row=0, column=0, sticky="e")
        self.diameter_entry = tk.Entry(self.circle_frame, width=10)
        self.diameter_entry.insert(0, "100")
        self.diameter_entry.grid(row=0, column=1, padx=5)

        tk.Label(settings_frame, text="Smoothing Level:").grid(row=6, column=0, sticky="e")
        self.smoothing_cb = ttk.Combobox(settings_frame, values=["Low", "Medium", "High"])
        self.smoothing_cb.set("Low")
        self.smoothing_cb.grid(row=6, column=1, padx=5)

        generate_btn = tk.Button(self.root, text="Generate STL", command=self.generate_lithophane)
        generate_btn.pack(pady=10)

    def on_shape_changed(self, event=None):
        if self.plate_shape.get() == "Rectangle":
            self.circle_frame.grid_remove()
            self.rectangle_frame.grid()
        else:  # Circle
            self.rectangle_frame.grid_remove()
            self.circle_frame.grid()
        
        # Update preview if image is loaded
        if self.original_image:
            self.update_3d_preview()

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Choose an image", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.heic")])
        if file_path:
            self.selected_image_path = file_path
            try:
                image = Image.open(file_path)
                image.thumbnail((300, 300))
                self.original_image = image.convert('L')

                photo = ImageTk.PhotoImage(self.original_image)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo

                self.update_3d_preview()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def update_3d_preview(self):
        if not self.original_image:
            return

        try:
            max_height = float(self.max_height_entry.get())
            pixel_scale = float(self.pixel_scale_entry.get())
            back_thickness = float(self.thickness_cb.get())
            
            # Process image based on selected shape
            if self.plate_shape.get() == "Rectangle":
                length = float(self.length_entry.get())
                width = float(self.width_entry.get())
                
                # Resize image to fit dimensions
                target_width = int(width / pixel_scale)
                target_height = int(length / pixel_scale)
                preview_img = self.original_image.resize((target_width, target_height), Image.LANCZOS)
                
                data = np.array(preview_img.transpose(Image.FLIP_TOP_BOTTOM))
                smoothed = gaussian_filter(data, sigma=1.0)
                heightmap = (1.0 - (smoothed / 255.0)) * max_height + back_thickness
                
                rows, cols = heightmap.shape
                x = np.linspace(0, width, cols)
                y = np.linspace(0, length, rows)
                xx, yy = np.meshgrid(x, y)
                
                fig = plt.Figure(figsize=(4.5, 3.5), dpi=100)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(xx, yy, heightmap, cmap='gray', edgecolor='none')
                ax.set_axis_off()
                
            else:  # Circle
                diameter = float(self.diameter_entry.get())
                radius = diameter / 2
                size = int(diameter / pixel_scale)
                
                # Resize and crop image to fit within a circle
                img_width, img_height = self.original_image.size
                aspect_ratio = img_width / img_height
                
                if aspect_ratio > 1:
                    new_width = size
                    new_height = int(size / aspect_ratio)
                else:
                    new_height = size
                    new_width = int(size * aspect_ratio)
                
                resized_img = self.original_image.resize((new_width, new_height), Image.LANCZOS)
                
                # Create a square image with the circle in the center
                square_img = Image.new('L', (size, size), 255)
                offset = ((size - new_width) // 2, (size - new_height) // 2)
                square_img.paste(resized_img, offset)
                
                # Create a circular mask
                mask = Image.new('L', (size, size), 0)
                mask_draw = ImageOps.invert(mask)
                center = size // 2
                for y in range(size):
                    for x in range(size):
                        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                        if dist <= center:
                            mask_draw.putpixel((x, y), 255)
                
                # Apply mask to image
                circular_img = Image.new('L', (size, size), 255)
                circular_img.paste(square_img, (0, 0), mask)
                
                data = np.array(circular_img.transpose(Image.FLIP_TOP_BOTTOM))
                smoothed = gaussian_filter(data, sigma=1.0)
                heightmap = (1.0 - (smoothed / 255.0)) * max_height + back_thickness
                
                rows, cols = heightmap.shape
                x = np.linspace(-radius, radius, cols)
                y = np.linspace(-radius, radius, rows)
                xx, yy = np.meshgrid(x, y)
                
                # Create a circular mask for the preview
                center_x = cols // 2
                center_y = rows // 2
                distance_from_center = np.sqrt((xx - 0)**2 + (yy - 0)**2)
                circle_mask = distance_from_center <= radius
                
                # Set points outside the circle to NaN for the preview
                heightmap_masked = heightmap.copy()
                heightmap_masked[~circle_mask] = np.nan
                
                fig = plt.Figure(figsize=(4.5, 3.5), dpi=100)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(xx, yy, heightmap_masked, cmap='gray', edgecolor='none')
                ax.set_axis_off()

            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            self.canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update preview: {e}")

    def generate_lithophane(self):
        if not self.original_image:
            messagebox.showwarning("No image", "Please select an image first.")
            return

        try:
            max_height = float(self.max_height_entry.get())
            pixel_scale = float(self.pixel_scale_entry.get())
            back_thickness = float(self.thickness_cb.get())
            
            save_path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])
            if not save_path:
                return
            
            if self.plate_shape.get() == "Rectangle":
                length = float(self.length_entry.get())
                width = float(self.width_entry.get())
                generate_rectangular_stl_from_image(self.original_image, max_height, pixel_scale, back_thickness, 
                                                 length, width, save_path)
            else:  # Circle
                diameter = float(self.diameter_entry.get())
                generate_circular_stl_from_image(self.original_image, max_height, pixel_scale, back_thickness, 
                                              diameter, save_path)
                
            messagebox.showinfo("Success", f"STL file saved to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate STL: {e}")

# ─── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Lithophane app as Flask server or Tkinter GUI')
    parser.add_argument('--gui', action='store_true', help='Launch the Tkinter GUI instead of the Flask server')
    parser.add_argument('--host', default='127.0.0.1', help='Flask host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Flask port (default: 5000)')
    args = parser.parse_args()

    if args.gui:
        root = tk.Tk()
        gui_app = LithophaneApp(root)
        root.mainloop()
    else:
        app.run(debug=True, host=args.host, port=args.port)