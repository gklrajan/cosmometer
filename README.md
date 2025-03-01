# Cosmo-Meter: Cosmic Event Detector

## Overview
**Cosmo-Meter** is an open-source cosmic event detection tool that uses digital camera sensors (CMOS or CCD) to detect **muons, cosmic rays, and other high-energy particle interactions**. By analyzing a time series of dark frames, it identifies transient bright pixel events that are likely caused by charged particles passing through the sensor. Use at least a 1000 dark images. The more, the better.

## What Can It Detect?
- **Most Likely Detections:**
  - **Muons** (from cosmic ray showers)
  - **High-energy protons and electrons**
  - **Secondary cosmic rays from air showers**

- **Possible Detections:**
  - **Radioactive decay events (beta particles, X-rays) in specific environments**
  - **High-energy gamma-ray interactions producing electron-positron pairs**
  - **Neutron-induced ionization (rare, but possible)**

- **What It Cannot Detect:**
  - **Neutrinos**
  - **Dark matter or exotic physics (unless they interact in an unknown way)**

## How It Works
Cosmo-Meter processes a sequence of dark frames (images taken in complete darkness) and:
1. **Identifies transient bright pixels** that appear in a single frame but disappear in subsequent frames.
2. **Detects streaks** (possible cosmic ray tracks across multiple pixels).
3. **Filters out hot pixels** (sensor defects that remain bright across many/ 0.5% frames).
4. **Logs detected cosmic events** with their coordinates and timestamps.

## Installation
### Requirements:
- Python 3.x
- Required Libraries:
  ```bash
  pip install numpy opencv-python matplotlib tifffile
  ```

## Usage
### Capture Dark Frames
1. Use a digital camera (CMOS/CCD) with the lens covered.
2. Capture a series of **dark frames** at a fixed exposure time (e.g., 30ms to 1s).
3. Save them in a folder (e.g., `dark_frames/`).

### Run Cosmo-Meter
```bash
python cosmometer.py --image_folder path/to/dark_frames --output cosmo-meter_events.csv
```


### Interpretation of Results
- The script outputs:
  - **Total detected cosmic particle events**
  - **Streaks (tracks across multiple pixels)**
  - **Muon flux per pixel per second**
  - **If sensor dimensions are provided, flux per mm² per second**
  
- A CSV file (`cosmo-meter_events.csv`) logs:
  - **Frame Number, X Coordinate, Y Coordinate** of each detected cosmic event.
  
- Graphical outputs include:
  - **Event heatmaps** (hot pixels, streaks, total cosmic events overlaid on sensor data)
  - **Scatter plot of detected cosmic events across frames**

## Example Output



## Contribute
- **Fork the repo & submit pull requests** to improve detection algorithms.
- **Citizen Science: test in different environments** (mountains, airplanes, underground labs) and share data. Use     Discussions to share your data.
- **Extend detection for specific particle types** (e.g., neutron sensitivity, radiation source analysis).

## License
Cosmos-Meter is released under the **MIT License**, making it open for anyone to use, modify, and improve.

## Credits
Developed by **Gokul Rajan** as an open-source cosmic event detection tool.

---
**The universe is always watching. Now, you can watch it back.** :D
