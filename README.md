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
