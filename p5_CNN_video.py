import os
import csv
from PIL import Image
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Conv2D
import matplotlib.pyplot as plt
import glob
import cv2
from keras.utils import plot_model
from keras.models import load_model
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


model = load_model('model_1.h5')

class HotMap:
    def __init__(self):
        self.last_10_box_list = []
        self.current_box_list = []
        self.frame_counter = 0

c_HotMap = HotMap()
# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, window_list, scale=1, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    xy_window = (int(xy_window[0] * scale), int(xy_window[1] * scale))
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to

    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def search_windows(img, windows):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        test_img = np.expand_dims(test_img, axis=0)
        prediction = model.predict(test_img, batch_size=None, verbose=0, steps=None)
        # print(prediction)

        if prediction > 0.5:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


####################################################################################################################

def process_frame(frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #draw_image = np.copy(frame)
    # model = load_model('model.h5')
    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    window_list = []

    y_start_stop = [400, 656]

    scale = 1.2
    window_list = slide_window(frame,
                               window_list,
                               scale=scale,
                               x_start_stop=[None, None],
                               y_start_stop=y_start_stop,
                               xy_window=(64, 64),
                               xy_overlap=(0.6, 0.6))
    scale = 1.5
    window_list = slide_window(frame,
                               window_list,
                               scale=scale,
                               x_start_stop=[None, None],
                               y_start_stop=y_start_stop,
                               xy_window=(64, 64),
                               xy_overlap=(0.6, 0.6))

    scale = 1.8
    window_list = slide_window(frame,
                               window_list,
                               scale=scale,
                               x_start_stop=[None, None],
                               y_start_stop=y_start_stop,
                               xy_window=(64, 64),
                               xy_overlap=(0.6, 0.6))

    box_list = search_windows(frame, window_list)
    c_HotMap.current_box_list = box_list

    if c_HotMap.frame_counter < 10:
        c_HotMap.last_10_box_list.append(box_list)
    else:
        for ii in range(10):
            if ii < 9:
                box_list = box_list + c_HotMap.last_10_box_list[ii]
                c_HotMap.last_10_box_list[ii] = c_HotMap.last_10_box_list[ii + 1]
            elif ii == 9:
                box_list = box_list + c_HotMap.last_10_box_list[ii]
                c_HotMap.last_10_box_list[ii] = c_HotMap.current_box_list

    window_img = draw_boxes(np.copy(frame), box_list, color=(0, 0, 255), thick=6)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 28)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)
    if c_HotMap.frame_counter < 10:
        c_HotMap.frame_counter = c_HotMap.frame_counter + 1
    return draw_img


###################################################################################################################
# Video

white_output = 'test_output.mp4'
#clip1 = VideoFileClip("project_video.mp4").subclip(18, 37)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_frame)  # NOTE: this function expects color images!!

white_clip.write_videofile(white_output, audio=False)
