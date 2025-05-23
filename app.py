import streamlit as st
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

st.title("🍅 Tomato Yield Estimation from Segmentation Mask")

# --- Your notebook functions (adapted for Streamlit) ---
import numpy as np
import cv2
import matplotlib.pyplot as plt

def color_match(mask, color, tolerance=10):
    return np.all(np.abs(mask - color) <= tolerance, axis=-1)

def generate_class_masks(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    ripe_mask = np.zeros_like(mask)
    semiripe_mask = np.zeros_like(mask)
    unripe_mask = np.zeros_like(mask)

    # Apply color filtering with some tolerance
    ripe_mask[color_match(mask, [0, 0, 255])] = [0, 0, 255]      # Blue
    semiripe_mask[color_match(mask, [255, 255, 0])] = [255, 255, 0]  # Yellow
    unripe_mask[color_match(mask, [0, 255, 0])] = [0, 255, 0]     # Green

    # Convert to BGR for saving
    ripe_mask_bgr = cv2.cvtColor(ripe_mask, cv2.COLOR_RGB2BGR)
    semiripe_mask_bgr = cv2.cvtColor(semiripe_mask, cv2.COLOR_RGB2BGR)
    unripe_mask_bgr = cv2.cvtColor(unripe_mask, cv2.COLOR_RGB2BGR)
    
    return ripe_mask, semiripe_mask, unripe_mask

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

def count_and_measure_tomatoes(
    mask_rgb,
    min_area=200,
    min_peak_distance=15,
    show_output=True
):
    # Load and preprocess
    #mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Distance transform + smoothing
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (3, 3), 0)

    # Detect local maxima
    local_max_coords = peak_local_max(
        distance,
        min_distance=min_peak_distance,
        labels=binary,
        footprint=np.ones((3, 3))
    )
    local_max_mask = np.zeros_like(distance, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    # Watershed segmentation
    markers = ndi.label(local_max_mask)[0]
    labels = watershed(-distance, markers, mask=binary)

    # Filter and measure regions
    final_labels = np.zeros_like(labels)
    sizes = []
    label_id = 1
    for l in range(1, np.max(labels) + 1):
        region = (labels == l).astype(np.uint8)
        area = np.sum(region)
        if area >= min_area:
            final_labels[labels == l] = label_id
            sizes.append(area)
            label_id += 1

    # Visualization
    if show_output:
        output = mask_rgb.copy()
        for label in range(1, np.max(final_labels) + 1):
            mask = np.uint8(final_labels == label)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (255, 0, 0), 2)

        plt.imshow(output)
        plt.title(f"Tomatoes: {len(sizes)} | Avg Size: {np.mean(sizes):.1f} px²")
        plt.axis("off")
        plt.show()

    return len(sizes), final_labels, sizes

def estimate_tomato_weights(areas, count, image_width, image_height, base_calibration_factor=0.04):
    # Calculate the total area of the image
    total_image_area = image_width * image_height

    # Ensure the areas are non-zero by checking the sum of the areas
    total_area = sum(areas)
    # if total_area == 0:
    #     raise ValueError("Total area of tomatoes cannot be zero")

    # Normalize the areas by the total area (or simply use raw areas)
    normalized_areas = [area / total_area for area in areas]

    # Adjust the calibration factor based on the number of tomatoes
    if count < 5:  # Assuming low count indicates tomatoes are closer/bigger
        calibration_factor = base_calibration_factor * 1.2  # Increase factor for larger tomatoes
    elif count > 20:  # Assuming high count indicates tomatoes are farther/smaller
        calibration_factor = base_calibration_factor * 0.8  # Decrease factor for smaller tomatoes
    else:
        calibration_factor = base_calibration_factor  # Keep base factor for moderate counts

    # Calculate individual tomato weights (scaled with normalized area)
    weights = [round(area * calibration_factor, 2) for area in areas]

    # Calculate total weight
    total_weight = round(sum(weights), 2)

    return weights, total_weight

def estimate_yield():
  ripe_count,ripe_labels,ripe_sizes =count_and_measure_tomatoes(ripe_mask)
  ripe_weights,ripe_total = estimate_tomato_weights(ripe_sizes,ripe_count,iheight,iwidth)

  semiripe_count,semiripe_labels,semiripe_sizes =count_and_measure_tomatoes(semiripe_mask)
  semiripe_weights,semiripe_total = estimate_tomato_weights(semiripe_sizes,semiripe_count,iheight,iwidth)

  unripe_count,unripe_labels,unripe_sizes =count_and_measure_tomatoes(unripe_mask)
  unripe_weights,unripe_total = estimate_tomato_weights(unripe_sizes,unripe_count,iheight,iwidth)

  print(ripe_count,"-----",semiripe_count,"-----",unripe_count)
  print(ripe_total,"-----",semiripe_total,"-----",unripe_total)
  #print(ripe_weights,semiripe_weights,unripe_weights)

  return ripe_count,semiripe_count,unripe_count, ripe_total,semiripe_total,unripe_total




# --- Streamlit Interface ---

uploaded_file = st.file_uploader("Upload a segmentation mask image (RGB)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    mask_np = np.array(image)
    st.image(mask_np, caption="Uploaded Mask", use_column_width=True)

    with st.spinner("Generating class masks..."):
        ripe_mask, semiripe_mask, unripe_mask = generate_class_masks(mask_np)
    iheight, iwidth, _ = ripe_mask.shape
    ripe_count,semiripe_count,unripe_count, ripe_total,semiripe_total,unripe_total=estimate_yield()
    st.image(ripe_mask)
    st.image(semiripe_mask)
    # Display all class masks
    st.subheader("🔹 Ripe Tomatoes")
    st.image(ripe_mask, caption="Ripe Mask", use_column_width=True)
    st.write(f"Count: {ripe_count}")
    st.write(f"Yield: {ripe_total} g")

    st.subheader("🟡 Semi-Ripe Tomatoes")
    st.image(semiripe_mask, caption="Semiripe Mask", use_column_width=True)
    st.write(f"Count: {semiripe_count}")
    st.write(f"Yield: {semiripe_total} g")

    st.subheader("🟢 Unripe Tomatoes")
    st.image(unripe_mask, caption="Unripe Mask", use_column_width=True)
    st.write(f"Count: {unripe_count}")
    st.write(f"Yield: {unripe_total} g")

    st.subheader("📊 Total Yield Summary")
    
    st.write("### 📊 Current Observations")
    st.write(f"**Ripe:** {ripe_count} | **Semi-Ripe:** {semiripe_count} | **Unripe:** {unripe_count}")
    st.write(f"**Ripe Total:** {ripe_total} | **Semi-Ripe Total:** {semiripe_total} | **Unripe Total:** {unripe_total}")
    
    st.write("### 🌾 Yield Estimation")
    st.write("**Yield ready to harvest:**", round(ripe_total / 1000, 2), "kilograms")
    st.write("**Yield ready to harvest in a week:**", round(semiripe_total * 1.1 / 1000, 2), "kilograms")
    st.write("**Yield ready to harvest in a month:**", round(unripe_total * 1.3 / 1000, 2), "kilograms")


