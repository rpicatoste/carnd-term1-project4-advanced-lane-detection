#**Self-Driving Car Nanodegree** 
##**Project 4 - Advance Lane Finding** 
**Ricardo Picatoste**

##Writeup 

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

In this document I consider the rubric points individually and describe how I addressed each point in my implementation.  

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 
This document.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `camera_calibration.py`.  

In the camera calibration python module I put the code to calibrate the camera from the images provided, and also I put the functions and variables related to the images transformation that will be used by the other modules of the project. The file can be imported, and it will provide the mentioned functions for the different transformations. If the module is run independently, the camera calibration and a function testing the different functions will automatically run using the condition:

	if __name__ == '__main__':
    calibrate_camera()
    sample_to_find_transformation_points()   
 
This is a very convenient way to have modules that can be imported and an automatic testing in case that it is required.

The camera calibration uses the images provided with the project. Those images are read and the points of the chess board found with the functions presented in the lessons. The images are stored in the output_images folder, and below an example is shown:

![alt text](./output_images/corners_found9.jpg "Corners found example")

From these points the distortion matrices are calculated, and also a matrix for the image transformation is obtained, with the result presented below. The following image is undistorted and warped with the obtained matrices.

![alt text]('./output_images/example_undistort_and_unwarp.png')

Finally, the distortion matrices to be used during the project are stored in a pickle file, `camera_calibration.p`, for easy use.

The variables and functions what will be used for the image transformations when searching for the lane lines are also explored in this module. Helper functions are provided to transform the images. Below the 2 sets of points used for the transformation are shown. One of the will be selected for the generation of the video, since it showed a final better result. 

For the first set of points:

![alt text]('./output_images/example_image_transformation_1_1.png')

![alt text]('./output_images/example_image_transformation_1_2.png')
 
And for the second, reaching less far from the car:

![alt text]('./output_images/example_image_transformation_2_1.png')
![alt text]('./output_images/example_image_transformation_2_2.png')
 

---
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The distortion and the transformation of images was described in the previous point.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the module `color_and_gradient.py`, all the functions related to color and gradient transformations are implemented. The module can be run independently, like with the camera module, and all the test performed while looking for the best combination of transforms to find the lane lines will be run; or it can also be included as module, giving access to all those functions.

The pipeline that seemed to work better to find the lane lines was a combination of thresholds in the saturation and lightness. It allowed to find both the white and the yellow lines. Below the pipeline found using this combination is used in an example.

![alt text]('./output_images/color_pipeline_example.png')
 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform was explained in the point related to the camera, where all the image corrections and transforms are dealt with.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions related to the lines search and fitting, and to the curvature and off-center distance calculations, are in the module `lanes.py`. Once again, the module can be executed independently, running some tests and showing the result. 

To detect the lane line, the proposed methodology has been followed. First, the position of the lines at the bottom of the image is found from the histogram of the bottom half of the image. From there, we go up and in there are more points from the binary mask to one side or the other, the rectangle considered is shifted to the right or to the bottom. Once the first fit is done, the next ones are done starting from the fit of the previous one instead of using the histogram. 

Below an example of the result using this methodology is shown: 

![alt text]('./output_images/detect_lane_pass_1.png')
 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature and the off-center distance are calculates in the lanes module. 

For the curvature, the formula proposed in the course has been used. This method consist of, once the second order polynomial has been fit to the datapoints, applying the derivative to the curve at the bottom point of the image. The only difference, trying to make it more robust, has been to calculate the curvature for the bottom 100 points of the lane lines, not just the last one, and using the average of them. Then a temporal moving average is also applied to smooth its value. In the previous point an example with the calculation done is presented.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In the main module, `p4_main.py`, the pipeline is implemented. This pipeline will take an image and will follow the steps described to find the lane position, the curvature of the road, and the position of the car respect to the center of the road.

![alt text]('./output_images/full_pipeline_example.png')

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The video with the result is included in the results folder of the repo.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The detection does no always have the same quality. To avoid spureous values and smooth the output, I created moving average and applied it to the radius and to the fitting coefficients. This improved those areas like changes in road color where the detection go give back wrong values. 