# Light Direction Estimation from Shadows

This project estimates light direction in images by analyzing the relationship between objects and their shadows. It use single stage instance shadow detection with bidirectional learning(based on the implemntation of Detectron2) for instance segmentation and Depth Anything for depth estimation.

## Features

- Detect objects and their shadows in images
- Estimate 3D light direction vectors
- Generate 3D visualizations with Open3D
- Export 3D models in PLY, OBJ, and GLB formats

## Installation

1. Clone this repository:
  
   git clone https://github.com/Luel-Abrha/light-direction-estimation-Deeplearning-project

**
Note: the all_in_one (updated at a later time)  source code contains all codes in one  and handles multiple shadows within an image and draws all the detected light direction in 3d(open3d). once  you  run it,  you will see the reconstructed object in 3d with open3d along with all detected light direction. you will find an Example of screenshut taken from the  open3d(capture 1 in output).
the other souce file handles only a single shadow even if there are multiple shadows. the light direction is also drawn in 2d image(light direction in 2d Image)**

## Running
python -m main.py
