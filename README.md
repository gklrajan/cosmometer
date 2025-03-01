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
![Screenshot 2025-03-01 at 07 43 42](https://github.com/user-attachments/assets/fdcc0b1a-17d7-4db5-9d60-0a23f788a94d)


### Interpretation of Results
The script outputs:  
- Total detected cosmic particle events  
- Streaks (tracks across multiple pixels)  
- Cosmic event flux per pixel per second -> updtaed to per day
- If sensor dimensions are provided, flux per mm² per second -> updated to per day
![Screenshot 2025-03-01 at 08 30 46](https://github.com/user-attachments/assets/30159d5c-a7e1-43d0-b097-a69b838a9748)



A **CSV file** (`cosmometer_events.csv`) logs:  
- Frame Number  
- X Coordinate  
- Y Coordinate  
For each detected cosmic event.
![Screenshot 2025-03-01 at 07 48 06](https://github.com/user-attachments/assets/926e4b79-8fff-495e-98a3-4f54bebac4dc)


Graphical outputs include:  

1. **Event heatmaps**  
   - Hot pixels (persistent bright pixels that could be artifacts)  
   - Streaks (tracks from high-energy cosmic rays)  
   - Total detected cosmic events overlaid on the sensor data
     ![Screenshot 2025-03-01 at 07 44 02](https://github.com/user-attachments/assets/5a23bde7-52e8-481d-84b4-e5952b8de075)


2. **Scatter plot of detected cosmic events over time**  
   - X, Y position of events  
   - Frame number as colorbar (shows when they occurred over time)
     ![Screenshot 2025-03-01 at 07 47 45](https://github.com/user-attachments/assets/5dd270e6-785e-4abb-96b6-d500bd08b5ca)


3. **Streak map visualization**  
   - Time (frame number) on X-axis  
   - Y-axis as streak coordinate (tracks cosmic rays over time)
     ![Screenshot 2025-03-01 at 07 47 21](https://github.com/user-attachments/assets/52168d34-43b6-4496-a716-09e7fe488843)



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
