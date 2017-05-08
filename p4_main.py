import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

import camera_calibration as cc


#%% Undistort and transform

# Get camera matrices
camera_dict = pickle.load( open('camera_calibration.p', mode='rb') )
mtx = camera_dict["mtx"] 
dist = camera_dict["dist"] 


#
fname = '.\CarND-Advanced-Lane-Lines\camera_cal\calibration3.jpg'
img = cv2.imread(fname)

top_down, perspective_M = cc.corners_unwarp(img, nx, ny, mtx, dist)

f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 7))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image')
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


