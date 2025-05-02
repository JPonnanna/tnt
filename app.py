import streamlit as st
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("🍅 Tomato Yield Estimation from Segmentation Mask")

# --- Your notebook functions (adapted for Streamlit) ---

def generate_class_masks(mask):
    ripe_mask = np.zeros_like(mask)
    semiripe_mask = np.zeros_like(mask)
    unripe_mask = np.zeros_like(mask)

    ripe_mask[np.all(mask == [255, 0, 0], axis=-1)] = [255, 0, 0]       # Red in BGR = Blue in RGB
    semiripe_mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 255, 255]  # Yellow
    unripe_mask[np.all(mask == [0, 255, 0], axis=-1)] = [0, 255, 0]     # Green

    return ripe_mask, semiripe_mask, unripe_mask

def count_and_measure_tomatoes(mask_np, min_area=200, min_peak_distance=15):
    mask_rgb = mask_np
    gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (3, 3), 0)

    local_max_coords = peak_local_max(
        distance,
        min_distance=min_peak_distance,
        labels=binary,
        footprint=np.ones((3, 3))
    )
    local_max_mask = np.zeros_like(distance, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    markers = ndi.label(local_max_mask)[0]
    labels = watershed(-distance, markers, mask=binary)

    sizes = []
    for l in range(1, np.max(labels) + 1):
        area = np.sum(labels == l)
        if area >= min_area:
            sizes.append(area)

    return sizes

def estimate_tomato_weights(areas):
    weights = [round(0.0023 * area, 2) for area in areas]
    return weights

def estimate_yield(weights):
    return round(sum(weights), 2)


# --- Streamlit Interface ---

uploaded_file = st.file_uploader("Upload a segmentation mask image (RGB)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    mask_np = np.array(image)
    st.image(mask_np, caption="Uploaded Mask", use_column_width=True)

    with st.spinner("Generating class masks..."):
        ripe_mask, semiripe_mask, unripe_mask = generate_class_masks(mask_np)

    # Display all class masks
    st.subheader("🔹 Ripe Tomatoes")
    st.image(ripe_mask, caption="Ripe Mask", use_column_width=True)
    ripe_areas = count_and_measure_tomatoes(ripe_mask)
    ripe_weights = estimate_tomato_weights(ripe_areas)
    st.write(f"Count: {len(ripe_areas)}")
    st.write(f"Weights: {ripe_weights}")
    st.write(f"Yield: {estimate_yield(ripe_weights)} kg")

    st.subheader("🟡 Semi-Ripe Tomatoes")
    st.image(semiripe_mask, caption="Semi-Ripe Mask", use_column_width=True)
    semiripe_areas = count_and_measure_tomatoes(semiripe_mask)
    semiripe_weights = estimate_tomato_weights(semiripe_areas)
    st.write(f"Count: {len(semiripe_areas)}")
    st.write(f"Weights: {semiripe_weights}")
    st.write(f"Yield: {estimate_yield(semiripe_weights)} kg")

    st.subheader("🟢 Unripe Tomatoes")
    st.image(unripe_mask, caption="Unripe Mask", use_column_width=True)
    unripe_areas = count_and_measure_tomatoes(unripe_mask)
    unripe_weights = estimate_tomato_weights(unripe_areas)
    st.write(f"Count: {len(unripe_areas)}")
    st.write(f"Weights: {unripe_weights}")
    st.write(f"Yield: {estimate_yield(unripe_weights)} kg")

    st.subheader("📊 Total Yield Summary")
    total_yield = (
        estimate_yield(ripe_weights) +
        estimate_yield(semiripe_weights) +
        estimate_yield(unripe_weights)
    )
    st.write(f"**Total Estimated Yield: {total_yield} kg**")

    # Optional Pie Chart
    st.subheader("🍰 Yield Distribution by Ripeness")
    labels = ['Ripe', 'Semi-Ripe', 'Unripe']
    values = [
        estimate_yield(ripe_weights),
        estimate_yield(semiripe_weights),
        estimate_yield(unripe_weights)
    ]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig)
