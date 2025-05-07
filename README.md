# Lithophane Generator (GUI Version)

A simple desktop application for generating 3D printable lithophanes from images using a clean and user-friendly GUI.

This project is built with Python and Tkinter, aiming to make lithophane generation accessible without the need for a browser or command line.

---

## ✨ Features

- 🖼️ Image selection and live preview
- 📎 Support for JPEG, PNG, HEIC formats
- 🧱 (Coming Soon) Convert image to 3D grayscale heightmap
- 📦 (Planned) Export to STL format
- 🎛️ (Planned) Customization options (thickness, smoothing, scaling)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/JayRKay91/lithophane-gui.git
cd lithophane-gui
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Run the App
bash
Copy
Edit
python app.py
🧰 Requirements
Python 3.9+

Pillow (image processing)

[Optional: numpy, scipy, trimesh for STL export]

📌 Roadmap
 Basic GUI and image loading

 Grayscale conversion preview

 STL generation

 Export settings and controls

 Packaged .exe release

📸 What Is a Lithophane?
A lithophane is a 3D-printed photo created by varying the thickness of a surface so that light passes through to form an image — commonly used in lamps, window hangers, and night lights.

🧑‍💻 Author
Created by JayRKay91

📝 License
This project is licensed under the MIT License. Feel free to fork and build upon it!

yaml
Copy
Edit

Here’s a quick step-by-step to get your v2 code up on the lithophane-gui-V2 branch and add a new README:

Make sure you’re on the right branch

bash
Copy
Edit
git checkout lithophane-gui-V2
Replace/update your code with the v2 files
Copy your updated app.py, any changed modules or templates, etc., into the working directory.

Create (or update) your README
If you want a completely new file, you might call it README_v2.md (so you don’t overwrite the existing one), e.g.:

bash
Copy
Edit
cat > README_v2.md <<EOF
# Lithophane Generator GUI — Version 2

**What’s new in v2:**
- Live image preview
- Adjustable height & thickness sliders
- STL export preset profiles
- …etc.

**Installation**  
```bash
pip install -r requirements.txt
python app.py
Usage

Click “Select Image”

Tweak your settings

Hit “Generate STL”
EOF
---

Once added, you can commit it:

```bash
git add README.md
git commit -m "Added project README"
git push