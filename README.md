# <span style="color:deepskyblue"> Real-time Object Detection and Tracking with YOLOv8, YOLOv11 & Streamlit </span>

This repository is an extensive open-source project showcasing the seamless integration of **object detection and tracking** using **YOLOv8, YOLOv11** (object detection algorithm), along with **Streamlit** (a popular Python web application framework for creating interactive web apps) and **OpenVINO**. The project offers a user-friendly and customizable interface designed to detect and track objects in real-time video streams from sources such as RTSP, UDP, and YouTube URLs, as well as static videos and images.


## <span style="color:deepskyblue">Explore Implementation Details on Medium (3 parts blog series) </span>
For a deeper dive into the implementation, check out my three-part blog series on [Medium](https://medium.com/@mycodingmantras), where I detail the [step-by-step process of creating this web application](https://medium.com/@mycodingmantras/building-a-real-time-object-detection-and-tracking-app-with-yolov8-and-streamlit-part-1-30c56f5eb956).


## <span style="color:deepskyblue">WebApp Demo on Streamlit Server</span>

Thank you team [Streamlit](<https://github.com/streamlit/streamlit>) for the community support for the cloud upload. 

This app is up and running on Streamlit cloud server!!! You can check the demo of this web application on this link 
[yolov8-streamlit-detection-tracking-webapp](https://yolov8-object-detection-and-tracking-app.streamlit.app/)

**Note**: In the demo, Due to non-availability of GPUs, you may encounter slow video inferencing.


## <span style="color:deepskyblue"> Tracking With Object Detection Demo</span>

<https://user-images.githubusercontent.com/104087274/234874398-75248e8c-6965-4c91-9176-622509f0ad86.mov>

## Overview

<https://github.com/user-attachments/assets/85df351a-371c-47e0-91a0-a816cf468d19.mov>


## Demo Pics

### Home page

<img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/pic1.png" >

### Page after uploading an image and object detection

<img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/pic3.png" >

### Segmentation task on image

<img src="https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/blob/master/assets/segmentation.png" >

## Requirements

Python 3.6+
YOLOv8
Streamlit
etc... Use requirements.txt
```bash
pip install -r requirements.txt
```

## Installation

*Ubuntu packages:*
sudo apt-get install ffmpeg libgl1-mesa-dev

*Windows*
Install ffmpeg and add it to system path

*Setup venv:*
```bash
conda create -n demo python=3.11

conda activate demo
```

*Clone repo, etc:*
- Clone the repository: `git clone https://github.com/js333031/yolov8-streamlit-detection-tracking.git`
- Change to the repository directory: `cd yolov8-streamlit-detection-tracking`
- Install requirements: `pip install -r requirements.txt`
```bash
git clone https://github.com/js333031/yolov8-streamlit-detection-tracking.git
cd yolov8-streamlit-detection-tracking
pip install -r requirements.txt
```
- Download the pre-trained YOLOv8 weights from (<https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt>) and save them to the `weights` directory in the same project.
- The application can also download the models and convert to OpenVINO format through the UI. When a model name is selected, the model will be downloaded and converted to OpenVINO format

## Usage

- Run the app with the following command: `streamlit run app.py`
- The app should open in a new browser window.
- To disable scrolling detection results:
```bash
Windows:
set YOLO_VERBOSE=False
streamlit run app.py

Linux:
export YOLO_VERBOSE=False
streamlit run app.py
```

### DL Model Config

- Select a model - if the model is not downloaded, it should get downloaded and converted to OpenVINO format. Check the console output
- Select task (Detection, Segmentation)
- Select model confidence
    - Use the slider to adjust the confidence threshold (25-100) for the model.
- Select inference engine (Ultralytics or OpenVINO backend within Ultralytics)
- Select inference engine device (CPU or Intel GPU) - used if inference engine is OpenVINO


One the model config is done, select a source.

### Detection on images

- The default image with its objects-detected image is displayed on the main page.
- Select a source. (radio button selection `Image`).
- Upload an image by clicking on the "Browse files" button.
- Click the "Detect Objects" button to run the object detection algorithm on the uploaded image with the selected confidence threshold.
- The resulting image with objects detected will be displayed on the page. Click the "Download Image" button to download the image.("If save image to download" is selected)

## Detection in Videos

- Create a folder with name `videos` in the same directory
- Dump your videos in this folder
- In `settings.py` edit the following lines.

```python
        # Videos config
        self.VIDEO_DIR = self.ROOT / 'videos'
        self.VIDEOS_DICT = {
            'video_1': self.VIDEO_DIR / 'video_1.mp4',
            'video_2': self.VIDEO_DIR / 'video_2.mp4',
            'video_3': self.VIDEO_DIR / 'video_3.mp4',
        }
```

- Click on `Detect Video Objects` button and the selected task (detection/segmentation) will start on the selected video.

### Detection on RTSP

- Select the RTSP stream button
- Enter the rtsp url inside the textbox and hit `Detect Objects` button

### Detection on YouTube Video URL

- Select the source as YouTube
- Copy paste the url inside the text box.
- The detection/segmentation task will start on the YouTube video url

<https://user-images.githubusercontent.com/104087274/226178296-684ad72a-fe5f-4589-b668-95c835cd8d8a.mov>

## Acknowledgements
[yolov8-streamlit-detection-tracking](https://github.com/rampal-punia/yolov8-streamlit-detection-tracking) was used as basis for this demo

This app uses [YOLOv8/11](<https://github.com/ultralytics/ultralytics>) for object detection algorithm and [Streamlit](<https://github.com/streamlit/streamlit>) library for the user interface.

### Disclaimer

This project is intended as a learning exercise and demonstration of integrating various technologies, including:

- Streamlit
- YoloV8
- Object-Detection on Images And Live Video Streams
- Python-OpenCV
- OpenVINO as backend to Ultralytics

Please note that this application is not designed or tested for production use. It serves as an educational resource and a showcase of technology integration rather than a production-ready web application.

Contributors and users are welcome to explore, learn from, and build upon this project for educational purposes.

### Hit star ⭐ if you like this repo!!!
