# Object Detection with OpenCSV

## Overview
This project uses various Python libraries to detect and track the movement of zebrafish in video footage. It analyzes videos to identify and count zebrafish, providing metrics such as recall, precision, and the approximate count of zebrafish.

## File Descriptions
- **Run.py**: Main script to process the video and output metrics.
- **common.py**: Contains common functions and utilities.
- **video.py**: Handles video processing tasks.
- **tst_scene_render.py**: For rendering scenes in tests.
- **MOVEMENT.csv**: Data file containing movement details after the video is finished.
  
## Features
- **Movement Detection**: Tracks zebrafish movement in the video.
- **Metric Calculation**: Computes recall, precision, and estimates the number of detected zebrafish.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Pandas

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage
To run the project, use the following command:
```bash
python Run.py ShortVideo.mp4
```

## Running the Project
https://github.com/user-attachments/assets/b492181c-fd04-4f94-bf44-7530ba41628a

## Disclaimer
This project is provided for educational and personal use only. It should not be used as part of academic, professional, or work-related project submissions. There is no permit to use this work in any capacity that violates academic or professional integrity policies.
