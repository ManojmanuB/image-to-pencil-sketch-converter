import streamlit as st
import cv2
import numpy as np

# --- App Config ---
st.set_page_config(page_title="Pro Pencil Sketch App", page_icon="🎨")
st.title("🎨 Pro Pencil Sketch & Webcam Sketch")

# --- Sidebar Settings ---
st.sidebar.title("Settings")
sketch_type = st.sidebar.radio("Sketch Type", ["Grayscale", "Color"])
style = st.sidebar.radio("Style", ["Pencil", "Color Pencil"])
blur_val = st.sidebar.slider("Blur Intensity", 1, 99, 21, step=2)  # Always odd for safety
contrast_val = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, step=0.1)
sharpen_val = st.sidebar.slider("Sharpen Strength", 0.0, 3.0, 1.0, step=0.1)

# --- Utility Function to Make Odd ---
def make_odd(val):
    val = max(1, val)
    return val if val % 2 == 1 else val + 1

# --- Pencil Portrait Sketch ---
def pencil_sketch(img, blur_val, contrast_val, sharpen_val):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # smooth skin/details

    inv = 255 - gray
    ksize = make_odd(blur_val)
    blur = cv2.GaussianBlur(inv, (ksize, ksize), 0)

    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # Adaptive contrast for natural details
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sketch = clahe.apply(sketch)

    # Adjust overall contrast
    sketch = cv2.convertScaleAbs(sketch, alpha=contrast_val, beta=0)

    # Sharpen based on slider
    if sharpen_val > 0:
        kernel = np.array([[0, -1, 0],
                           [-1, 4 + sharpen_val, -1],
                           [0, -1, 0]])
        sketch = cv2.filter2D(sketch, -1, kernel)

    return sketch

# --- Color Pencil Sketch ---
def color_pencil_sketch(img, blur_val, contrast_val, sharpen_val):
    # Create pencil sketch first
    sketch_gray = pencil_sketch(img, blur_val, contrast_val, sharpen_val)

    # Convert sketch to 3 channels
    sketch_color = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

    # Blend with original image for color pencil effect
    blended = cv2.multiply(sketch_color / 255.0, img / 255.0)
    blended = np.uint8(blended * 255)
    return blended

# --- Cartoon Sketch ---
def cartoon_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

# --- Main Sketch Function ---
def create_sketch(img):
    if style == "Pencil":
        sketch = pencil_sketch(img, blur_val, contrast_val, sharpen_val)
    elif style == "Color Pencil":
        sketch = color_pencil_sketch(img, blur_val, contrast_val, sharpen_val)
    elif style == "Cartoon":
        sketch = cartoon_sketch(img)

    # If user chose Color mode for Pencil, adapt it
    if sketch_type == "Color" and style == "Pencil":
        sketch_color = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        sketch = cv2.multiply(sketch_color / 255.0, img / 255.0)
        sketch = np.uint8(sketch * 255)

    return sketch

# --- Tabs: Upload / Webcam ---
tab1, tab2 = st.tabs(["Upload Image", "Webcam Sketch"])

# --- Upload Image Tab ---
with tab1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        sketch_img = create_sketch(img)

        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
        col2.image(cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB), caption=f"{style} Sketch", use_column_width=True)

        _, buffer = cv2.imencode(".png", sketch_img)
        st.download_button("Download Sketch", data=buffer.tobytes(), file_name="sketch.png", mime="image/png")

# --- Webcam Tab (Smooth) ---
with tab2:
    stframe = st.empty()
    run = st.checkbox("Start Webcam")
    
    if run:
        cap = cv2.VideoCapture(0)
        capture = st.button("Capture Snapshot")
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not access webcam.")
                break

            frame = cv2.flip(frame, 1)
            sketch_frame = create_sketch(frame)

            # Show live sketch
            stframe.image(cv2.cvtColor(sketch_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

            # Snapshot button
            if capture:
                _, buffer = cv2.imencode(".png", sketch_frame)
                st.download_button("Download Snapshot", data=buffer.tobytes(),
                                   file_name="webcam_sketch.png", mime="image/png")
                capture = False

        cap.release()
