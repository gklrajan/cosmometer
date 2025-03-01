import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import tifffile as tiff
import csv
from tqdm import tqdm


def load_images(image_folder):
    """Loads a sequence of images from a folder, supporting multiple formats."""
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.*")))  # Match all file types
    images = []

    for f in image_files:
        try:
            if f.lower().endswith(".tif") or f.lower().endswith(".tiff"):
                img = tiff.imread(f)
            else:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {f} ({e})")

    return np.array(images), image_files


def compute_mean_image(images):
    """Computes the mean image across the entire stack."""
    mean_image = np.mean(images, axis=0)
    return mean_image


def detect_particle_events(images, mean_image, hot_pixel_thresh=0.005, duration_thresh=10, streak_connectivity=3):
    """Detects cosmic events, hot pixels, and streaks based on a pixel-wise adaptive threshold."""
    num_frames, height, width = images.shape
    event_map = np.zeros((height, width), dtype=np.uint32)
    hot_pixel_map = np.zeros((height, width), dtype=np.uint32)
    streak_map = np.zeros((num_frames, height, width), dtype=np.uint32)  # Stores streaks across time

    threshold_matrix = 5 * mean_image  # Pixel-wise threshold

    event_list = []
    activation_count = np.zeros((height, width), dtype=np.uint32)
    active_duration = np.zeros((height, width), dtype=np.uint32)

    # Analyze frame differences
    for i in tqdm(range(num_frames), desc="Processing frames"):
        events = images[i] > threshold_matrix
        event_map += events.astype(np.uint32)
        activation_count += events  # Count how often each pixel is active

        # Track consecutive activations
        active_duration[events] += 1
        active_duration[~events] = 0  # Reset if pixel turns off

        # Store detected events
        for y, x in zip(*np.where(events)):
            event_list.append((i, x, y))

        # Streak detection: Require at least 3 connected pixels
        for y, x in zip(*np.where(events)):
            if np.sum(events[max(0, y-1):y+2, max(0, x-1):x+2]) >= streak_connectivity:
                streak_map[i, y, x] = 1

    # Hot pixel filtering: Remove pixels active in >0.5% of total frames
    hot_pixels = activation_count > (hot_pixel_thresh * num_frames)
    event_map[hot_pixels] = 0

    # Remove pixels that were ON for more than `duration_thresh` consecutive frames
    long_active_pixels = active_duration > duration_thresh
    event_map[long_active_pixels] = 0

    return event_map, hot_pixels, streak_map, event_list


def save_events_to_csv(event_list, output_csv):
    """Saves detected cosmic particle events to a CSV file."""
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "X Coordinate", "Y Coordinate"])
        writer.writerows(event_list)
    print(f"Particle event data saved to {output_csv}")


def plot_results(event_map, hot_pixels, streak_map, event_list):
    """Plots detected cosmic particle events, hot pixels, streaks, and event locations."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].imshow(event_map, cmap='hot', interpolation='none')
    ax[0].set_title("Cosmic Particle Events")

    ax[1].imshow(hot_pixels, cmap='gray', interpolation='none')
    ax[1].set_title("Detected Hot Pixels")

    plt.show()

    # Streak visualization: y on color bar, time on x-axis
    if np.any(streak_map):
        streak_y, streak_x, streak_t = np.where(streak_map)
        plt.figure(figsize=(10, 5))
        plt.scatter(streak_t, streak_x, c=streak_y, cmap='plasma', marker='o', alpha=0.5)
        plt.xlabel("Frame Number")
        plt.ylabel("X Coordinate")
        plt.colorbar(label="Y Coordinate")
        plt.title("Detected Cosmic Streaks Across Time")
        plt.show()

    # Scatter plot for cosmic event locations
    if event_list:
        event_frame, event_x, event_y = zip(*event_list)
        plt.figure(figsize=(10, 5))
        plt.scatter(event_x, event_y, c=event_frame, cmap='viridis', marker='o', alpha=0.5)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.colorbar(label="Frame Number")
        plt.title("Detected Cosmic Particle Events")
        plt.show()


def main(image_folder, output_csv, pixel_size_um=None, sensor_width_px=None, sensor_height_px=None, exposure_time_ms=1, binning_factor=1):
    """Cosmo-Meter: Detecting cosmic rays, muons, and high-energy particle interactions in digital camera sensors."""
    images, image_files = load_images(image_folder)
    print(f"Loaded {len(images)} images from {image_folder}")

    mean_image = compute_mean_image(images)
    event_map, hot_pixels, streak_map, event_list = detect_particle_events(images, mean_image)

    total_particle_events = np.sum(event_map)
    total_streaks = np.sum(streak_map)

    print(f"Total detected cosmic particle events: {total_particle_events}")
    print(f"Total detected streaks: {total_streaks}")

    save_events_to_csv(event_list, output_csv)
    plot_results(event_map, hot_pixels, streak_map, event_list)


# Run the script
if __name__ == "__main__":
    main(
        image_folder="D:/Gokul/2024-Widefield/dark_img_allCovered_",
        output_csv="cosmo-meter_events.csv",
        pixel_size_um=6.5,
        sensor_width_px=2048,
        sensor_height_px=1024,
        exposure_time_ms=30,
        binning_factor=1
    )
