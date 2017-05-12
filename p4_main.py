
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
global counter, left_fit, right_fit
counter = 0
left_fit, right_fit = None, None

def pipeline_image( image, first_frame = False ):
    
    global counter
    if first_frame:
        counter = 0
    else:
        counter += 1
        
    result_img,_,_ = pipeline( image, counter )
    return result_img

def pipeline( image, counter = 0, plot_image = False):
    
    global left_fit, right_fit
    
    # undistort and unwarp
    warped = cc.undistort_and_warp(image)
    warped_binary = cg.pipeline_filter_to_binary(warped, s_thresh = threshold_s, l_thresh = threshold_l, plot_layers = False )
    
    if(counter == 1):
        y, left_x, left_y, left_fit, right_fit = lanes.fit_lane_to_binary_image( warped_binary )
    else:
        y, left_x, left_y = lanes.fit_lane_to_binary_image_from_previos_fit( warped_binary, left_fit, right_fit )
  
    
    left_rad, right_rad     = lanes.get_curvature(y, left_x, left_y)
        
    
    result = lanes.fill_found_lanes_in_original(image, y, left_x, left_y)
    
    cv2.putText( result, 'It: {}'.format(counter), 
                 (120,140), fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    
    if( plot_image ):    
        f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))
        f.tight_layout()
        ax1.imshow(result)
        ax1.set_title(image_file)
        ax2.imshow(warped_binary, cmap = 'gray')
        ax2.set_title('left_rad ' + str(left_rad) + ', right_rad ' + str(right_rad) )
    
    return result, left_rad, right_rad
  
      

images = glob.glob(r'.\CarND-Advanced-Lane-Lines\test_images\test*.jpg')

images.append( r'.\CarND-Advanced-Lane-Lines\test_images\straight_lines1.jpg' )  
images.append( r'.\CarND-Advanced-Lane-Lines\test_images\straight_lines2.jpg' )  

threshold_s = (150, 255)
threshold_l = (210, 255)

for image_file in images:
    image = cv2.imread( image_file )
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    result,_,_ = pipeline(image, True)
            


import sys;sys.exit("Cricho exit")

#%% Test one after another

## Imagen problematica
##image_file1 = r'.\CarND-Advanced-Lane-Lines\test_images\test5_prev.jpg'
##image_file2 = r'.\CarND-Advanced-Lane-Lines\test_images\test5.jpg'
#
#image_file1 = r'.\CarND-Advanced-Lane-Lines\test_images\test6.jpg'
#image_file2 = r'.\CarND-Advanced-Lane-Lines\test_images\test6.jpg'
#
#image1 = cv2.imread( image_file1 )
#image1 = cv2.cvtColor( image1, cv2.COLOR_BGR2RGB )
#    
#image2 = cv2.imread( image_file2 )
#image2 = cv2.cvtColor( image2, cv2.COLOR_BGR2RGB )
#
## undistort and unwarp
#warped1 = cc.undistort_and_warp(image1)
#warped_binary1 = cg.pipeline_filter_to_binary(warped1, s_thresh = (150, 255), l_thresh = (210, 255), plot_layers = False )
#y, left_x, left_y, left_fit, right_fit = lanes.fit_lane_to_binary_image( warped_binary1, plot_figure = False )
#
#warped2 = cc.undistort_and_warp(image2)
#warped_binary2 = cg.pipeline_filter_to_binary(warped2, s_thresh = (150, 255), l_thresh = (210, 255), plot_layers = False )
#y, left_x, left_y = lanes.fit_lane_to_binary_image_from_previos_fit( warped_binary2, left_fit, right_fit, plot_figure = False )
##y, left_x, left_y, left_fit, right_fit = lanes.fit_lane_to_binary_image( warped_binary2, plot_figure = False )
#
#left_rad, right_rad = lanes.get_curvature(y, left_x, left_y)
#
#
#%% Videoooo

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os 

directory = os.path.dirname("results/")
if not os.path.exists(directory):
    os.makedirs(directory)
        
#%%
# next video
videos = [ 'project_video', 'challenge_video', 'harder_challenge_video' ]
videos = [ 'harder_challenge_video' ]
project_video_result_file = 'results\\'

for video in videos:
    if('clip' in vars() or 'clip' in globals()):
        del clip
    clip = VideoFileClip( os.path.join('CarND-Advanced-Lane-Lines', video + '.mp4') )
    project_video_result = clip.fl_image(pipeline_image) #NOTE: this function expects color images!!
    #%time 
    project_video_result.write_videofile('results\\' + video + '_result.mp4', audio=False)
    print("Video " + video + " done")
    