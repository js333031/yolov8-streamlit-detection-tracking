from pathlib import Path
import os
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video_1': VIDEO_DIR / 'video_1.mp4',
    'video_2': VIDEO_DIR / 'video_2.mp4',
    'video_3': VIDEO_DIR / 'video_3.mp4',
}

MODEL_NAMES = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL_FILENAME = 'yolov8n'
DETECTION_MODEL = Path.joinpath(MODEL_DIR,DETECTION_MODEL_FILENAME + '.pt')
DETECTION_MODEL_OV = Path.joinpath(MODEL_DIR, DETECTION_MODEL_FILENAME + "_openvino_model", DETECTION_MODEL_FILENAME + '.xml')
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

SEGMENTATION_MODEL_FILENAME = 'yolov8n-seg'
SEGMENTATION_MODEL = Path.joinpath(MODEL_DIR,SEGMENTATION_MODEL_FILENAME + '.pt')
SEGMENTATION_MODEL_OV = Path.joinpath(MODEL_DIR, SEGMENTATION_MODEL_FILENAME + "_openvino_model", DETECTION_MODEL_FILENAME + '.xml')

# Webcam
WEBCAM_PATH = 0

def update_model_names(NEW_MODEL_NAME):
    DETECTION_MODEL_FILENAME = NEW_MODEL_NAME
    DETECTION_MODEL = Path.joinpath(MODEL_DIR,DETECTION_MODEL_FILENAME + '.pt')
    DETECTION_MODEL_OV = Path.joinpath(MODEL_DIR, DETECTION_MODEL_FILENAME + "_openvino_model", DETECTION_MODEL_FILENAME + '.xml')
    # In case of your custome model comment out the line above and
    # Place your custom model pt file name at the line below 
    # DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

    SEGMENTATION_MODEL_FILENAME = 'yolov8n-seg'
    SEGMENTATION_MODEL = Path.joinpath(MODEL_DIR,SEGMENTATION_MODEL_FILENAME + '.pt')
    SEGMENTATION_MODEL_OV = Path.joinpath(MODEL_DIR, SEGMENTATION_MODEL_FILENAME + "_openvino_model", DETECTION_MODEL_FILENAME + '.xml')

def print_model_names():
    print(f"Model Root path: {ROOT}")
    fstat = os.path.isfile(DETECTION_MODEL)
    print(f"Ultralytics Detection model path: {DETECTION_MODEL}, file exists: {fstat}")

    fstat = os.path.isfile(SEGMENTATION_MODEL_OV)
    print(f"Ultralytics Segmentation model path: {SEGMENTATION_MODEL_OV}, file exists: {fstat}")

    fstat = os.path.isfile(DETECTION_MODEL_OV)
    print(f"OpenVINO Detection model path: {DETECTION_MODEL_OV}, file exists: {fstat}")

    fstat = os.path.isfile(SEGMENTATION_MODEL_OV)
    print(f"OpenVINO Segmentation model path: {SEGMENTATION_MODEL_OV}, file exists: {fstat}")

print_model_names()