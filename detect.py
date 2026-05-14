import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

IMAGES_DIR   = "images/"
ANNOTATIONS  = "annotations.csv"
VISUALIZE_DETECTED = 0

TRAY_THRESHOLD  = 80    # Pixels darker than this = tray (increase if tray not found)
LID_THRESHOLD   = 87    # Pixels brighter than this inside tray = lid (tune this most)
LID_MIN_AREA    = 400   # Ignore blobs smaller than this (noise)
LID_MAX_AREA    = 1600  # Ignore blobs larger than this (tray artifacts)
MATCH_THRESHOLD = 20    # Max pixel distance to count a prediction as correct
BLOCKSIZE = 277

 
def find_tray(cropped):
    """Find the bounding box of the dark tray in the cropped image."""
    gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
 
    _, dark_thresh = cv2.threshold(blurred, TRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    if not contours:
        return None
 
    # Tray is the largest dark region
    tray_contour = max(contours, key=cv2.contourArea)
    tx, ty, tw, th = cv2.boundingRect(tray_contour)

    # # Checking if adding padding to the bounding box gives better results
    # PADDING = 1  # pixels — increase if lids on edges are still cut off
    # h, w = cropped.shape[:2]
    # tx = max(0, tx - PADDING)
    # ty = max(0, ty - PADDING)
    # tw = min(w - tx, tw + 2 * PADDING)
    # th = min(h - ty, th + 2 * PADDING)

    return tx, ty, tw, th
 
def find_lids_in_tray(tray_crop):
    """Find lid blobs inside the tray crop. Returns list of (cx, cy, radius)."""
    gray    = cv2.cvtColor(tray_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
 
    # _, lid_thresh = cv2.threshold(blurred, LID_THRESHOLD, 255, cv2.THRESH_BINARY)
    lid_thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=BLOCKSIZE,   # local region size — must be odd, tune this
        C=-14           # lids must be this much brighter than local average
    )
 
    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    lid_thresh = cv2.morphologyEx(lid_thresh, cv2.MORPH_OPEN, kernel)
 
    contours, _ = cv2.findContours(lid_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(tray_crop, contours,0,0000)
    # cv2.imshow("contour",tray_crop)
    # cv2.waitKey(0)
    h, w = tray_crop.shape[:2]
    lids = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Check if contour touches the tray boundary
        x, y, cw, ch = cv2.boundingRect(contour)
        touches_edge  = (x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2)
        # Allow smaller area for edge lids since they are partially cut off
        min_area = LID_MIN_AREA // 2 if touches_edge else LID_MIN_AREA
        if area < LID_MIN_AREA or area > LID_MAX_AREA:
            continue
        if area < LID_MIN_AREA or area > LID_MAX_AREA:
            continue
 
        M = cv2.moments(contour)
        # if M["m00"] == 0:
        #     continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        radius = int(np.sqrt(area / np.pi))
        angle, ellipse  = estimate_angle(contour)  
        
        (ex, ey), _, _ = ellipse
        cv2.circle(tray_crop, (int(ex), int(ey)), 3, (255, 0, 0), -1)

        # # Drawing the ellipse to check if it is fitting good or not 

        # cv2.ellipse(tray_crop, ellipse, (0, 0, 255), 2)
        # (ex, ey), (major, minor), ellipse_angle = ellipse
        # angle_rad = np.deg2rad(ellipse_angle+90)
        # dx = np.cos(angle_rad) * major / 2
        # dy = np.sin(angle_rad) * major / 2
        # end1 = (int(ex + dx), int(ey + dy))
        # end2 = (int(ex - dx), int(ey - dy))
        # cv2.line(tray_crop, (int(ex), int(ey)), end1, (255, 255, 0), 2)  # cyan = one end
        # cv2.line(tray_crop, (int(ex), int(ey)), end2, (0, 165, 255), 2)
        # cv2.imshow("ellipse fit", tray_crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        lids.append((float(cx), float(cy), float(radius), float(angle)))
 
    return lids
  
def detect_circles_tray_n_angles(image):
    """
    Full pipeline: crop → find tray → find lids inside tray.
    Returns list of (x, y, radius) in full image coordinates.
    """
    h, w = image.shape[:2]
 
    # Crop to top-right quarter
    crop_x = w // 2
    crop_y = 0
    cropped = image[crop_y : h // 2, crop_x : w]
 
    # Find tray
    tray = find_tray(cropped)
    # tray = find_tray(image)
    if tray is None:
        return []
 
    tx, ty, tw, th = tray
 
    # Crop to tray region
    tray_crop = cropped[ty : ty + th, tx : tx + tw]
 
    # Find lids inside tray
    lids = find_lids_in_tray(tray_crop)
 
    # Convert coordinates corresponding to cropped image to full image 
    detections = []
    for cx, cy, r, ang in lids:
        full_x = cx + tx +crop_x
        full_y = cy + ty +crop_y
        detections.append((full_x, full_y, r, ang))
 
    return detections

def estimate_angle(contour):
    """
    Fit an ellipse to the lid contour.
    The tab is the pointy end — find it using contour extreme points.
    Returns angle in degrees [0, 360).
    """
    if len(contour) < 5:
        return 0.0  # fitEllipse needs at least 5 points

    # Fit ellipse to get the main axis direction
    ellipse = cv2.fitEllipse(contour)
    (ex, ey), (major, minor), ellipse_angle = ellipse
#-----------------------------------------------------------------tried finding the tab end by trying to find 
# the pointier side of the ellipse, but this didnt give any better results-----------------

    # # ellipse_angle is the angle of the minor axis in degrees (0-180)
    # # Convert to a direction vector along the major axis
    # # adding 90 to convert minor axis angle to major axis angle 
    # angle_rad = np.deg2rad(ellipse_angle + 90)
    # dx = np.cos(angle_rad)
    # dy = np.sin(angle_rad)

    # # The two ends of the major axis
    # end1 = np.array([ex + dx * major / 2, ey + dy * major / 2])
    # end2 = np.array([ex - dx * major / 2, ey - dy * major / 2])

    # # The tab end is pointier — find which end is farther from the contour centroid
    # # by checking which end has less contour area around it (pointy = less mass)
    # M = cv2.moments(contour)
    # cx = M["m10"] / M["m00"]
    # cy = M["m01"] / M["m00"]

    # # Check which end is the tab by seeing which end the contour is narrower
    # # We do this by checking pixel distances from contour points to each end
    # contour_pts = contour.reshape(-1, 2).astype(np.float32)

    # dist_to_end1 = np.min(np.linalg.norm(contour_pts - end1, axis=1))
    # dist_to_end2 = np.min(np.linalg.norm(contour_pts - end2, axis=1))

    # # The tab end is the pointy tip — contour points are farther from it
    # # So the end with larger min distance to contour = the tab
    # if dist_to_end1 > dist_to_end2:
    #     tab_end = end1
    # else:
    #     tab_end = end2

    # # Angle = direction from center to tab
    # vec_x = tab_end[0] - cx
    # vec_y = tab_end[1] - cy
    # angle = np.degrees(np.arctan2(-vec_y, vec_x))  # negative y because y axis is flipped
    # angle = angle % 360
    # return angle, ellipse
#-----------------------------------------------------------------------------------------------
    # ellipse_angle is the angle of the minor axis in degrees (0-180)
    # adding 90 to convert minor axis angle to major axis angle 
    angle_rad = np.deg2rad(ellipse_angle + 90)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    angle = np.degrees(np.arctan2(-dy, dx))
    angle = angle % 360

    return angle, ellipse

def match_predictions_to_gt(pred_centers, gt_centers):
    """
    We are iteratively matching all predicted centres to the gt_centres that are not already matched, 
    we store the indices of the matched centres from the gt_centres in a set for this, for each amtch we increase the tp count
    and then using total predictions and total positives we calculate fp and fn respectively.
    """
    matched_gt = set()
    tp = 0

    for px, py in pred_centers:
        best_dist = float("inf")
        best_idx  = -1

        for i, (gx, gy) in enumerate(gt_centers):
            if i in matched_gt:
                continue
            dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        if best_dist <= MATCH_THRESHOLD and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp
    return tp, fp, fn

def circular_angle_error(pred_angle, gt_angle):
    # diff = abs(pred_angle - gt_angle) % 360
    # return min(diff, 360 - diff)

    """Absolute difference between predicted and truth values. If flipping pred by 180 gives less error, use that."""
    
    diff_original = abs(pred_angle - gt_angle) % 360
    diff_flipped  = abs((pred_angle + 180) - gt_angle) % 360

    return min( min(diff_original, 360 - diff_original)  ,   min(diff_flipped, 360 - diff_flipped) )

def show_image(image, gt_centers, detections, title):
    """Draw GT (green) and predictions (red) on image and show it."""
    vis = image.copy()
    for gx, gy in gt_centers:
        cv2.circle(vis, (int(gx), int(gy)), 5, (0, 255, 0), -1)
    for x, y, r, a in detections:
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 0, 255), 2)
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
    cv2.imshow(title, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def diagnose(image_path, gt_centers):
    """Show threshold image + contours with area labels to understand misses."""
    image = cv2.imread(image_path)
    h, w  = image.shape[:2]

    crop_x  = w // 2
    cropped = image[0 : h // 2, crop_x : w]

    gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    _, dark_thresh = cv2.threshold(blurred, TRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _    = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tray_contour   = max(contours, key=cv2.contourArea)
    tx, ty, tw, th = cv2.boundingRect(tray_contour)

    tray_crop = cropped[ty : ty + th, tx : tx + tw]
    tray_gray = cv2.cvtColor(tray_crop, cv2.COLOR_BGR2GRAY)
    tray_blur = cv2.GaussianBlur(tray_gray, (5, 5), 1)

    _, lid_thresh = cv2.threshold(tray_blur, LID_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel        = np.ones((3, 3), np.uint8)
    lid_thresh    = cv2.morphologyEx(lid_thresh, cv2.MORPH_OPEN, kernel)

    all_contours, _ = cv2.findContours(lid_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = tray_crop.copy()

    for contour in all_contours:
        area = cv2.contourArea(contour)
        M    = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if LID_MIN_AREA < area < LID_MAX_AREA:
            # Passing filter — green
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
            cv2.putText(vis, str(int(area)), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # Failing filter — red, with area label so you know why
            cv2.drawContours(vis, [contour], -1, (0, 0, 255), 2)
            cv2.putText(vis, str(int(area)), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw GT centers shifted into tray crop coordinates
    for gx, gy in gt_centers:
        gx_local = gx - crop_x - tx
        gy_local = gy - ty
        cv2.circle(vis, (int(gx_local), int(gy_local)), 6, (255, 255, 0), 2)  # yellow = GT

    # Also show the raw threshold image
    cv2.imshow("threshold", lid_thresh)
    cv2.imshow("contours + GT", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# function to save the images annotated with the detections from the script
# def save_detections(image_path, gt_centers, detections, output_dir="detections"):
#     """Save image with GT (yellow) and predictions (green/red) marked to output folder."""
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     image = cv2.imread(image_path)
#     vis   = image.copy()

#     # Draw GT centers in yellow
#     for gx, gy in gt_centers:
#         cv2.circle(vis, (int(gx), int(gy)), 8, (0, 255, 255), 2)

#     # Draw predictions — green if matched to a GT, red if not
#     pred_centers = [(x, y) for x, y, r, a in detections]
#     matched_gt   = set()
#     matched_pred = set()

#     for pred_idx, (px, py) in enumerate(pred_centers):
#         best_dist = float("inf")
#         best_idx  = -1
#         for i, (gx, gy) in enumerate(gt_centers):
#             if i in matched_gt:
#                 continue
#             dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
#             if dist < best_dist:
#                 best_dist = dist
#                 best_idx  = i
#         if best_dist <= MATCH_THRESHOLD and best_idx >= 0:
#             matched_gt.add(best_idx)
#             matched_pred.add(pred_idx)

#     for pred_idx, (px, py, r, a) in enumerate(detections):
#         color = (0, 255, 0) if pred_idx in matched_pred else (0, 0, 255)  # green = TP, red = FP
#         cv2.circle(vis, (int(px), int(py)), int(r), color, 2)
#         cv2.circle(vis, (int(px), int(py)), 3, color, -1)

#     filename    = os.path.basename(image_path)
#     output_path = os.path.join(output_dir, filename)
#     cv2.imwrite(output_path, vis)

def run():
    annotations = pd.read_csv(ANNOTATIONS)
    # image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])
    image_files = sorted([f for f in os.listdir(IMAGES_DIR)])

    total_tp, total_fp, total_fn = 0, 0, 0
    angle_errors = []

    for filename in image_files:
        image_path = os.path.join(IMAGES_DIR, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not load {filename}, skipping.")
            continue

        # Ground truth for this image
        gt_rows    = annotations[annotations["image"] == filename]
        gt_centers = list(zip(gt_rows["center_x"], gt_rows["center_y"]))
        gt_angles  = list(gt_rows["angle_deg"])

        # Detect circles
        detections   = detect_circles_tray_n_angles(image)
        pred_centers = [(x, y) for x, y, r, a in detections]

        # Match and compute TP/FP/FN
        tp, fp, fn = match_predictions_to_gt(pred_centers, gt_centers)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Compute angle error for matched predictions similarily as we calculated the 
        # True positives by comparing predicted and ground truth values
        matched_gt_set = set()
        for pred_idx, (px, py) in enumerate(pred_centers):
            best_dist = float("inf")
            best_idx  = -1
            for i, (gx, gy) in enumerate(gt_centers):
                if i in matched_gt_set:
                    continue
                dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i
            if best_dist <= MATCH_THRESHOLD and best_idx >= 0:
                matched_gt_set.add(best_idx)
                r          = detections[pred_idx][2]
                pred_angle = detections[pred_idx][3]
                gt_angle   = gt_angles[best_idx]
                angle_errors.append(circular_angle_error(pred_angle, gt_angle))

        print(f"{filename}: GT={len(gt_centers)} | Pred={len(pred_centers)} | TP={tp} FP={fp} FN={fn}")

        # # We wont visualize the images in this loop as outputting all the images will be heavy task, 
        # # instead will use run_single() to visualize whenever needed
        #   
        # if VISUALIZE_DETECTED:
        #     show_image(image, gt_centers, detections, filename)

    # ─── Print summary ────────────────────────────────────────────────────────
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    angle_mae = np.mean(angle_errors) if angle_errors else float("nan")

    print()
    print("*" * 50)
    print("EVALUATION SUMMARY")
    print("*" * 50)
    print(f"  Total images   : {len(image_files)}")
    print(f"  Total tubes : {total_tp + total_fn}")
    print(f"  TP , FP , FN   : {total_tp} , {total_fp} , {total_fn}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1 Score       : {f1:.4f}")
    print(f"  Angle MAE (degrees)  : {angle_mae} ")
    print("*" * 50)
    # checking the variation of the angle error values
    # print(f"Min angle error = {np.min(angle_errors)}")
    # print(f"Max angle error = {np.max(angle_errors)}")
    # print(f"Std deviation in angle error = {np.std(angle_errors)}")
    # plt.plot(angle_errors)
    # plt.xlabel("True positives")
    # plt.ylabel("angel error")
    # plt.show()

def run_single():
    ''' Function to run the detection on a single image, used to try out different methods adn viusalize images as the run()'''
    annotations = pd.read_csv(ANNOTATIONS)
    # image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])
    image_files = sorted([f for f in os.listdir(IMAGES_DIR)])

    total_tp, total_fp, total_fn = 0, 0, 0
    angle_errors = []

    for _ in range(1):
        # image_path = os.path.join(IMAGES_DIR, filename)
        filename = "71a6769b-color.png"
        image = cv2.imread(f"images/{filename}")

        if image is None:
            print(f"Could not load {filename}, skipping.")
            continue

        # Ground truth for this image
        gt_rows    = annotations[annotations["image"] == filename]
        gt_centers = list(zip(gt_rows["center_x"], gt_rows["center_y"]))
        gt_angles  = list(gt_rows["angle_deg"])

        # Detect circles
        detections = detect_circles_tray_n_angles(image)
        pred_centers = [(x, y) for x, y, r, a in detections]

        # print(gt_centers, "\n")
        # print(pred_centers)
    

        # Match and compute TP/FP/FN
        tp, fp, fn = match_predictions_to_gt(pred_centers, gt_centers)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        matched_gt_set = set()
        for pred_idx, (px, py) in enumerate(pred_centers):
            best_dist = float("inf")
            best_idx  = -1
            for i, (gx, gy) in enumerate(gt_centers):
                if i in matched_gt_set:
                    continue
                dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i
            if best_dist <= MATCH_THRESHOLD and best_idx >= 0:
                matched_gt_set.add(best_idx)
                r          = detections[pred_idx][2]
                pred_angle = detections[pred_idx][3]
                gt_angle   = gt_angles[best_idx]
                angle_errors.append(circular_angle_error(pred_angle, gt_angle))
                # print(pred_angle, gt_angle)
                

        print(f"{filename}: GT={len(gt_centers)} | Pred={len(pred_centers)} | Angle errors for TP={angle_errors} | TP={tp} FP={fp} FN={fn}")

        if VISUALIZE_DETECTED:
            show_image(image, gt_centers, detections, filename)

# run_single()
run()
