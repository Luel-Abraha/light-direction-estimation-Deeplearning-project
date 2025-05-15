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


