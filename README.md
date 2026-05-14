# Testtube_Lid_detection

Thought Peocess -
First tried using hough circles method to detect circles in the image and get the centre coordiantes and radius. But to shadows in the background and different edges present in various images it was able to identify the lids as circle, also the lids are not perfect circles, they are more oval shaped so this did not work.
There was one observation, in all the images the testtube tray is placed in the top right corner of the image, we can use this to crop images so that we get rid of the background noise, and to get the coordinates of the centre we can always just add the cropped amount. Added this one step and tried hough circles again but it was yet not able to capture the circles, so the major problem is with the shape and color of the lids itself.
So to tackle the color problem, after converting to gray scale, applied a threshold to covnert it into black n white image so that the lid shape is visible clearly. Also the hough circles was not going to work so had to try using contour finding method to find the lid shapes in the black n white image. It was having problem with this method because the background around the test tube tray was also mostly converted to white completely, because of this the lids on the edge merged into the background and there were mis detections. So to solve this what we could do is we will crop the image further to only the tray, this can be done easily because the tray is dark coloured. We will find the four corners of the tray and then create a box with some width around it so to capture the lids on the edge as well. Then we find the countour in the tray cropped, black n white image, now it is easier to find them. After finding the contour we filter them based on their areas, that is we will accept contour that are within a particular range only this will ensure we only take the contour that are probably detecting the lids, the value for this upper and lower limit was decide observing the various values for the true positives lids. After this we find the

after getting the contour we use m00, m10, m01 to find the centroid of the the shape that will be our centre of circle. m00 gives the area, m01 and m10 give the sum of all y and x coordinates of the points in the contour, for radius we calculate a value based on the area of the countour so technically all our circles are a bit larger than they should be because the from observing we can see that the lid circles should be inscribes in the oval like shape of the lid (the joint ant tap make it look like oval). But we can go ahead with this because detecting centres more accuratley is more important tha the radius.

the detect_circles_tray() function combines this entire pipeline into one function, it uses the find_lids_in_tray(tray_cropped_image), find_tray(cropped_image) functions internally, 

then for each image we calculate the number of true positives, FP, FN. and then add these for all the images to calculate the precision, recall, f1 score
for each image We are iteratively matching all predicted centres to the gt_centres that are not already matched, we store the indices of the matched centres from the gt_centres in a set for this, for each amtch we increase the tp count and then using total predictions and total positives we calculate fp and fn respectively.

then in the main run() function we run a loop over all image files from the folder and get the total TP, FP, FN to calculate the metrics.

TRAY_THRESHOLD
LID_THRESHOLD     
LID_MIN_AREA     
LID_MAX_AREA    
MATCH_THRESHOLD

are the major tunable parameters, currently we have a lot of FN in the outcome, so next aim is to improve the number of detections and make sure they are right, we dont want more of FP. And the angle detection part remains as well 

For estimating the angle, the estimate_angle() function takes the contour as input and fits ellipse to them and then gives the angle of the major axis and the ellipse objest as output. The find_lids_in_tray() function calls this function to calculate the angle along with the centre coordinated of the circle fit to the lid, it appends all these values in a list 'lids' and returns it, the function detect_circles_tray_n_angles() calls this find_lids_in_tray(), modifies the coordinated of the centre according to the full image and appends all the values in the detections array and returns that. In the main run() function we get the detections from the detect_circles_tray_n_angles() function and then calculate the angle errors using the circular_angle_error() function and append these values to the angle_errors array and report the mean of this as the MAE error in angle detection.


increased the max area thresh to 1600 this increased the tp and reduced fn and fp. From observing the detections found out that in many images the tray detection is failing badly because fo tray not boing dark enough due to lighting in the image, many lids are getting cut fromt he edges so they are being detected due to leaa area maybe, reducing the min area thresh might increase noise too so we can just do that, A lot of images didnt have any lids in them because the tray detection failed, especially when the background of the tray is black. other case was tray not dark enough so the thresh was proving to be more for it and lids were not deferentiated from tray.

To solve the problem of the threshold being fixed and the tray not having a fixed pixel value, we can use OTSU's algorithm that gives adaptive threshold,
Otsu's algorithm looks at the histogram of pixel values in the image and automatically finds the threshold that best separates dark pixels from bright pixels. So for a bright image it might pick 120, for a dark image it might pick 60 — it adapts per image. The 0 you pass in is ignored, Otsu computes its own value.
Now for the lids that were being cut off from the edge, what we can do is have a separte threshold for their areas, for this first we need to check whether the contour is near the edge or not for this cv2.boundingRect gives you the rectangle around the contour. We check if any side of that rectangle is within 2 pixels of the tray crop boundary — if yes, the lid is at the edge and likely cut off. If it's an edge lid, we halve the minimum area requirement. So a partially cut lid with area <400 still passes. Normal lids still use the full LID_MIN_AREA so noise doesn't increase for interior detections.
next the lids not getting thresholded due to lighting, Same problem as before — 90 is a fixed global value. If the tray is not very dark (poor lighting), the tray pixels might be at value 85 and the lids at 95 — very close together, so threshold 90 either misses lids or includes tray.

lid_thresh = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=21,
    C=-5
)
Instead of one global threshold, adaptive thresholding divides the image into small regions (blockSize=21 means 21x21 pixel patches) and computes a local threshold for each region based on the average brightness of that patch.
So if one corner of the tray is brighter due to lighting, it computes a higher threshold there. If another corner is darker, it uses a lower threshold there. Each region adapts to its own local brightness.
The C=-5 is a constant subtracted from the local average. It means "a pixel must be at least 5 brightness units above the local average to be considered bright". Negative C makes it more inclusive — without it, half the pixels in every region would always be "bright" which is too noisy. We ran a loop and checked what value of blocksize gives best result, we found the the peak in f1 score at 277, so that was chosen as its value.
After this I ran the main function for different values of C from -20 to 10 to find te best result and -14 gives highest f1 score.