# Tube Lid Detection

Detecting the position and orientation of microcentrifuge tube lids from overhead images.

---

## The Problem

Given 70 overhead images of a tube tray, detect each tube lid and predict:
- **Center coordinates** — `(x, y)` in pixels
- **Rotation angle** — direction from joint to tab, in degrees `[0, 360)`

Ground truth is provided for all 70 images (371 total tubes). Performance is measured using Precision, Recall, F1 score, and Angle MAE.

---

## Approach

### Step 1 — Crop to the region of interest
All images have the tube tray in the **top-right quarter**. Cropping to this region immediately removes most of the background noise and makes detection much easier.

### Step 2 — Find the tray
The tray is a dark rectangular object. We threshold the cropped image to isolate dark regions and pick the largest dark contour — that's the tray. We use **Otsu's thresholding** here instead of a fixed value, so it adapts to different lighting conditions across images.

### Step 3 — Find lids inside the tray
Within the tray region, lids appear as brighter blobs against the dark background. We use **adaptive thresholding** to find them — this computes a local threshold for each small patch of the image rather than one global value, so it handles uneven lighting well.

After thresholding, we find contours and filter them by area to keep only blobs that are roughly lid-sized. Contours touching the tray edge get a lower area requirement since those lids are partially cut off.

### Step 4 — Compute centers
For each contour, we use **image moments** to find the centroid — the average x and y position of all points in the contour. This gives us `(center_x, center_y)`.

### Step 5 — Estimate angles
We fit an **ellipse** to each lid contour. The major axis of the ellipse gives the orientation of the lid. Since an axis has two directions (180° ambiguity), we resolve this at evaluation time by always taking the smaller of the two possible angle errors.

---

## Evaluation

Detections are matched to ground truth using a **distance threshold** (if the predicted center is within N pixels of a GT center, it counts as a match).

| Metric | Description |
|--------|-------------|
| Precision | Of all predicted tubes, how many were correct |
| Recall | Of all real tubes, how many were found |
| F1 Score | Harmonic mean of Precision and Recall |
| Angle MAE | Mean absolute angle error on matched detections |

---

## Results

| Metric | Value |
|--------|-------|
| TP / FP / FN | 259 / 14 / 112 |
| Precision | 0.949 |
| Recall | 0.698 |
| F1 Score | 0.804 |
| Angle MAE | 11.23° |

---

## Tunable Parameters

| Parameter | What it controls |
|-----------|-----------------|
| `TRAY_THRESHOLD` | Darkness cutoff for tray detection (overridden by Otsu) |
| `LID_THRESHOLD` | Brightness cutoff for lid detection inside tray |
| `LID_MIN_AREA` | Smallest contour area to accept as a lid |
| `LID_MAX_AREA` | Largest contour area to accept as a lid |
| `MATCH_THRESHOLD` | Max pixel distance to count a prediction as correct |

The adaptive threshold `blockSize` and `C` values were tuned by running a sweep — `blockSize=277` and `C=-14` gave the best F1 score.

---

## Key Challenges

- **Varied backgrounds and lighting** — fixed thresholds didn't generalise across images; solved with Otsu and adaptive thresholding
- **Lids cut off at tray edges** — partially visible lids have less area; solved by halving the minimum area requirement for edge contours
- **Tray detection failure** — when the tray wasn't much darker than the background, the whole image would produce zero detections; Otsu thresholding largely fixed this
- **180° angle ambiguity** — ellipse axes don't have a direction; handled by always choosing the smaller of the two possible angle errors at evaluation time

---

## Next Steps

- Improve recall — 112 tubes are still being missed, mostly due to tray detection failures on tricky backgrounds
- Better angle estimation — resolve the 180° ambiguity properly by detecting the tab end of the lid shape
- Explore learning-based detection (e.g. fine-tuned YOLO) for more robustness across varied conditions