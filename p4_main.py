
import numpy as np
import cv2
import matplotlib.pyplot as plt

import camera_calibration as cc
import color_and_gradient as cg
import lanes

import glob

#%% Undistort and transform

## Get camera matrices
#camera_dict = pickle.load( open('camera_calibration.p', mode='rb') )
#mtx = camera_dict["mtx"] 
#dist = camera_dict["dist"] 
#
#fname = '.\CarND-Advanced-Lane-Lines\camera_cal\calibration3.jpg'
#img = cv2.imread(fname)
#
#top_down, perspective_M = cc.corners_unwarp(img, cc.nx, cc.ny, mtx, dist)
#
#f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 7))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image')
#ax2.imshow(top_down)
#ax2.set_title('Undistorted and Warped Image')
##plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#%%

images = glob.glob(r'.\CarND-Advanced-Lane-Lines\test_images\test*.jpg')

images.append( r'.\CarND-Advanced-Lane-Lines\test_images\straight_lines1.jpg' )  
images.append( r'.\CarND-Advanced-Lane-Lines\test_images\straight_lines2.jpg' )  

for image_file in images:# [r'.\CarND-Advanced-Lane-Lines\test_images\test2.jpg']:#
    image = cv2.imread( image_file )
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
        
    # undistort and unwarp
    warped = cc.undistort_and_warp(image)
    
    warped_binary = cg.pipeline(warped, s_thresh = (150, 255), l_thresh = (210, 255), plot_layers = False )
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title(image_file)
    ax2.imshow(warped_binary, cmap = 'gray')
    ax2.set_title('Mixed')
    
    y, left_x, left_y = lanes.fit_lane_to_binary_image( warped_binary, plot_figure = True )
    
    left_rad, right_rad = lanes.get_curvature(y, left_x, left_y)
    
    print('left_rad', left_rad, 'right_rad', right_rad)

#    hls_binary = hls_select(image, thresh=(90, 255), print_layers = True)


