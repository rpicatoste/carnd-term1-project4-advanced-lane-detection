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



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform was explained in the point related to the camera, where all the image corrections and transforms are dealt with.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
