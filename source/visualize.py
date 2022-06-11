# Copyright 2021 Toyota Research Institute.  All rights reserved.

from cv2 import COLOR_BGR2RGB
import numpy as np
import sys
import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append("../camviz")
sys.path.append("../packnet-sfm")
import cv2
from packnet_sfm.datasets.augmentations import resize_image
from packnet_sfm.utils.depth import load_depth

from PIL import Image
import camviz as cv

# Load evaluation data
data_path = "../Kitti/3_out/"

depth = []
rgb = []
f = open("../Kitti/3_pose/pose.json")
pose = json.load(f)
intrinsics = np.loadtxt("../Kitti/3_pose/intrinsic.txt")

for filename in os.listdir(data_path):
    img = cv2.imread(data_path+filename)
    img = np.array(resize_image(Image.fromarray(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (384,1280)))
        
    depth.append(load_depth('../Kitti/3_depth/' + filename[:-3]+"npz"))
    rgb.append(np.array(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))

    

# Get image resolution
wh = rgb[0].shape[:2]
print(wh)
# Create draw tool with specific width and height window dimensions
draw = cv.Draw(wh=(2000, 900), title='CamViz Pointcloud Demo')

# Create image screen to show the RGB image
draw.add2Dimage('rgb', luwh=(0.00, 0.00, 1.00, 0.50),res=(1280,384))

# Create image screen to show the depth visualization
# draw.add2Dimage('viz', luwh=(0.00, 0.50, 0.33, 1.00), res=wh)

# Create world screen at specific position inside the window (% left/up/right/down)
draw.add3Dworld('wld', luwh=(0.00, 0.50, 1.00, 1.00))

# Parse dictionary information



# Create camera from intrinsics and image dimensions (width and height)
camera = cv.objects.Camera(K=intrinsics, wh=wh)

# Project depth maps from image (i) to camera (c) coordinates
points = [camera.i2c(depth[i]) for i in range(len(depth))]

# Create pointcloud colors
rgb_clr = [rgb[i].reshape(-1, 3) / 255 for i in range(len(rgb))]                   # RGB colors
# viz_clr = viz.reshape(-1, 3)                   # Depth visualization colors
# hgt_clr = cv.utils.cmaps.jet(-points[:, 1])    # Height colors

# Create RGB and visualization textures
for i in range(len(rgb)):
    draw.addTexture('rgb'+str(i), rgb[i] / 255 ) 
    draw.addBufferf('pts'+str(i), points[i])   # Create data buffer to store depth points
    draw.addBufferf('clr'+str(i), rgb_clr[i]) 

# Color dictionary
color_dict = {0: 'clr', 1: 'viz', 2: 'hgt'}

# Display loop
color_mode = 0
index = 0
while draw.input():

    # If RETURN is pressed, switch color mode
    if draw.RETURN:
        color_mode = (color_mode + 1) % len(color_dict)

    # Clear window
    draw.clear()
    # Draw image textures on their respective screens
    draw['rgb'].image('rgb'+str(index))
    # draw['viz'].image('viz')
    # Draw points and colors from buffer
    draw['wld'].size(2).points('pts'+str(index), 'clr'+str(index))

    # Draw camera with texture as image
    # draw['wld'].object(camera, tex='rgb')

    # Update window
    index += 1
    if index >= len(rgb):
        index = 0
    draw.update(60)
