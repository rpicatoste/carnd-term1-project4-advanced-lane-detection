
import numpy as np
import cv2
import matplotlib.pyplot as plt

import camera_calibration as cc
import color_and_gradient as cg
import lanes
from MovingAverage import MovingAverage

import glob

#%% Undistort and transform

## Get camera matrices
#camera_dict = pickle.load( open('camera_calibration.p', mode='rb') )
#mtx = camera_dict["mtx"] 
#dist = camera_dict["dist"] 
#
#fname = '.\camera_cal\calibration3.jpg'
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
global counter, left_rad_filt, right_rad_filt 
left_rad_filt  = MovingAverage(8)
right_rad_filt = MovingAverage(8)
left_fit_filt  = MovingAverage(6) 
right_fit_filt = MovingAverage(6)

counter = 0

def pipeline_image( image, force_first_frame = False, plot_figure = False ):
    
    global counter
    if force_first_frame:
        counter = 1
    else:
        counter += 1
        
    result_img,_,_ = pipeline( image, counter_in = counter, plot_image = plot_figure)
    
    return result_img


def pipeline( image, counter_in = 0, plot_image = False):
    
    global counter, left_fit_filt, right_fit_filt, left_rad_filt, right_rad_filt
    
    lane_width_pixels = 700 # Default value
    
    # undistort and unwarp
    warped = cc.undistort_and_warp(image)
    warped_binary = cg.pipeline_filter_to_binary(warped, 
                                                 s_thresh = threshold_s, 
                                                 l_thresh = threshold_l, 
                                                 plot_layers = plot_image )
    
    if(counter_in == 1):
        y, left_x, left_y, left_fit, right_fit, lane_width_pixels = \
                                        lanes.fit_lane_to_binary_image(warped_binary, 
                                                                       plot_figure = plot_image )
        if y is None:
            counter = 0
    else:
        y, left_x, left_y, left_fit, right_fit = \
                                        lanes.fit_lane_to_binary_image_from_previous_fit(warped_binary,
                                                                                         left_fit_filt.average,
                                                                                         right_fit_filt.average, 
                                                                                         plot_figure = plot_image )
    
    if y is not None :
        left_fit_filt.next_val( left_fit )
        right_fit_filt.next_val( right_fit )
    else:
        print('Liada con x!')
        return image, None, None
    
    left_rad_m, right_rad_m, off_center_m = lanes.get_curvature_and_off_center_distance(y, 
                                                                                        left_fit_filt.average,
                                                                                        right_fit_filt.average,
                                                                                        lane_width_pixels)
    
    left_rad_filt.next_val( left_rad_m )
    right_rad_filt.next_val( right_rad_m )
            
    result = lanes.fill_found_lanes_in_original(image, y,  left_fit_filt.average, right_fit_filt.average)
    
                
    if( plot_image ):   
        
        cv2.putText(result, 
                    'Radious left{: 6} m, right{: 6} m'.format( int(left_rad_filt.average), 
                                                                int(right_rad_filt.average) ), 
                    (120,60), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1, 
                    color=(0,0,0), 
                    thickness = 2 )
                    
        cv2.putText(result, 
                    'Distance off-center {:.2} m.'.format(off_center_m), 
                    (120,100), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1, 
                    color=(0,0,0), 
                    thickness = 2 ) 

        fig = plt.figure(dpi=60, figsize=(25, 10))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        fig.tight_layout()
 
        ax1.imshow(result)
        ax1.set_title('Result')
        ax2.imshow(warped_binary, cmap = 'gray')
        ax2.set_title('Warped')
                               
        fig.savefig(r'.\output_images\full_pipeline_example.png')
        
    return result, left_rad_m, right_rad_m
      

images = glob.glob(r'.\test_images\test*.jpg')

images.append( r'.\test_images\straight_lines1.jpg' )  
images.append( r'.\test_images\straight_lines2.jpg' )  

threshold_s = (150, 255)
threshold_l = (210, 255)

for image_file in images:
    left_fit_filt.restart()
    right_fit_filt.restart()
    image = cv2.imread( image_file )
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    result = pipeline_image(image, True, plot_figure = True)

import sys;sys.exit("Cricho exit")

#%% Videoooo

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import os 

directory = os.path.dirname("results/")
if not os.path.exists(directory):
    os.makedirs(directory)
        

# next video
videos = [ 'project_video', 'challenge_video', 'harder_challenge_video' ]
videos = [ 'recorte' ]
videos = [ 'project_video' ]
project_video_result_file = 'results\\'


for video in videos:
        
    counter = 0
    left_rad_filt.restart()
    right_rad_filt.restart()

    if('clip' in vars() or 'clip' in globals()):
        del clip
    clip = VideoFileClip( os.path.join('CarND-Advanced-Lane-Lines', video + '.mp4') )
    project_video_result = clip.fl_image( pipeline_image ) #NOTE: this function expects color images!!
 
    project_video_result.write_videofile('results\\' + video + '_result.mp4', audio=False)
    print("Video " + video + " done")
    