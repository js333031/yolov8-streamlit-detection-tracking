from pathlib import Path
import os
import sys
from ultralytics import YOLO

class Settings:
    def __init__(self, default_det_model_name, default_seg_model_name):
        # Get the absolute path of the current file
        self.FILE = Path(__file__).resolve()
        # Get the parent directory of the current file
        self.ROOT = self.FILE.parent
        # Add the root path to the sys.path list if it is not already there
        if self.ROOT not in sys.path:
            sys.path.append(str(self.ROOT))
        # Get the relative path of the root directory with respect to the current working directory
        self.ROOT = self.ROOT.relative_to(Path.cwd())

        # Sources
        self.IMAGE = 'Image'
        self.VIDEO = 'Video'
        self.WEBCAM = 'Webcam'
        self.RTSP = 'RTSP'
        self.YOUTUBE = 'YouTube'

        self.SOURCES_LIST = [self.IMAGE, self.VIDEO, self.WEBCAM, self.RTSP, self.YOUTUBE]

        # Images config
        self.IMAGES_DIR = self.ROOT / 'images'
        self.DEFAULT_IMAGE = self.IMAGES_DIR / 'office_4.jpg'
        self.DEFAULT_DETECT_IMAGE = self.IMAGES_DIR / 'office_4_detected.jpg'

        # Videos config
        self.VIDEO_DIR = self.ROOT / 'videos'
        self.VIDEOS_DICT = {
            'video_1': self.VIDEO_DIR / 'video_1.mp4',
            'video_2': self.VIDEO_DIR / 'video_2.mp4',
            'video_3': self.VIDEO_DIR / 'video_3.mp4',
        }

        self.MODEL_NAMES = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

        # ML Model config
        self.MODEL_DIR = self.ROOT / 'weights'
        self.DETECTION_MODEL_FILENAME = default_det_model_name
        self.DETECTION_MODEL = Path.joinpath(self.MODEL_DIR, self.DETECTION_MODEL_FILENAME + '.pt')
        self.DETECTION_MODEL_OV = Path.joinpath(self.MODEL_DIR, self.DETECTION_MODEL_FILENAME + "_openvino_model", self.DETECTION_MODEL_FILENAME + '.xml')
        # In case of your custome model comment out the line above and
        # Place your custom model pt file name at the line below 
        # DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

        self.SEGMENTATION_MODEL_FILENAME = default_seg_model_name
        self.SEGMENTATION_MODEL = Path.joinpath(self.MODEL_DIR, self.SEGMENTATION_MODEL_FILENAME + '.pt')
        self.SEGMENTATION_MODEL_OV = Path.joinpath(self.MODEL_DIR, self.SEGMENTATION_MODEL_FILENAME + "_openvino_model", self.DETECTION_MODEL_FILENAME + '.xml')

        # Webcam
        self.WEBCAM_PATH = 0

    def update_model_names(self, NEW_MODEL_NAME):
        self.DETECTION_MODEL_FILENAME = NEW_MODEL_NAME
        self.DETECTION_MODEL = Path.joinpath(self.MODEL_DIR,self.DETECTION_MODEL_FILENAME + '.pt')
        self.DETECTION_MODEL_OV = Path.joinpath(self.MODEL_DIR, self.DETECTION_MODEL_FILENAME + "_openvino_model", self.DETECTION_MODEL_FILENAME + '.xml')
        self.convert_modelto_ov(self.DETECTION_MODEL_OV, self.DETECTION_MODEL)
        # In case of your custome model comment out the line above and
        # Place your custom model pt file name at the line below 
        # DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

        self.SEGMENTATION_MODEL_FILENAME = NEW_MODEL_NAME+'-seg'
        self.SEGMENTATION_MODEL = Path.joinpath(self.MODEL_DIR, self.SEGMENTATION_MODEL_FILENAME + '.pt')
        self.SEGMENTATION_MODEL_OV = Path.joinpath(self.MODEL_DIR, self.SEGMENTATION_MODEL_FILENAME + "_openvino_model", self.SEGMENTATION_MODEL_FILENAME + '.xml')
        self.convert_modelto_ov(self.SEGMENTATION_MODEL_OV, self.SEGMENTATION_MODEL)

    def convert_modelto_ov(self, model_path, ul_model_path):
        if not model_path.exists():
            model = YOLO(ul_model_path)
            model.export(format="openvino", dynamic=True, half=False)

    def print_model_names(self):
        print(f"Model Root path: {self.ROOT}")
        fstat = os.path.isfile(self.DETECTION_MODEL)
        print(f"Ultralytics Detection model path: {self.DETECTION_MODEL}, file exists: {fstat}")

        fstat = os.path.isfile(self.SEGMENTATION_MODEL_OV)
        print(f"Ultralytics Segmentation model path: {self.SEGMENTATION_MODEL_OV}, file exists: {fstat}")

        fstat = os.path.isfile(self.DETECTION_MODEL_OV)
        print(f"OpenVINO Detection model path: {self.DETECTION_MODEL_OV}, file exists: {fstat}")

        fstat = os.path.isfile(self.SEGMENTATION_MODEL_OV)
        print(f"OpenVINO Segmentation model path: {self.SEGMENTATION_MODEL_OV}, file exists: {fstat}")
