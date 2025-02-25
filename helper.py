from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
#import settings
import openvino as ov
import os
from pathlib import Path

import time

class FpsTracker:
    def __init__(self):
        self.fps_start_time = time.time()
        self.fps=0
        self.fps_diff_time = 0
    
    def get_fps(self):
        self.fps_end_time = time.time()
        self.fps_diff_time = self.fps_end_time - self.fps_start_time
        self.fps = 1 / self.fps_diff_time
        self.fps_start_time = self.fps_end_time
        self.fps_text="FPS:{:.2f}".format(self.fps)
        return self.fps_text

fps_tracker = None

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def load_model_ov(model_path, device, task="detect"):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    ov_config = {}
    p=os.path.dirname(model_path)
    core = ov.Core()
    ov_model = core.read_model(model_path)
    print("Det model path: ", model_path)
    print("Using model path: ", p)
    if device != "CPU":
        ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    print(ov_config)
    print(f"Compiling OV model to {device}")
    ov_compiled_model = core.compile_model(ov_model, device, ov_config)

    ul_det_model = YOLO(model_path.parent, task=task)
    if ul_det_model.predictor is None:
        print("Configuring predictor")
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
        args = {**ul_det_model.overrides, **custom}
        ul_det_model.predictor = ul_det_model._smart_load("predictor")(overrides=args, _callbacks=ul_det_model.callbacks)
        ul_det_model.predictor.setup_model(model=ul_det_model.model)
    
    ul_det_model.predictor.model.ov_compiled_model = ov_compiled_model
    return ul_det_model

def convert_modelto_ov(model_path, ul_model_path):
    if not model_path.exists():
        model = load_model(ul_model_path)
        model.export(format="openvino", dynamic=True, half=False)

    ov_model = ov.Core().read_model(model_path)
    return ov_model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None, fps_tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf, verbose=False)

    if fps_tracker is not None:
        fps_text = fps_tracker.get_fps()
    else:
        fps_text = None

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    if fps_text is not None:
        cv2.putText(res_plotted, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    ret, buffer = cv2.imencode('.jpg', res_plotted, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if ret:
        st_frame.image(buffer.tobytes(),
                   caption='Detected Video',
                   channels="BGR",
                   use_container_width=True
                   )


def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'no_warnings': True,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def play_youtube_video(conf, model, settings):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        if not source_youtube:
            st.sidebar.error("Please enter a YouTube URL")
            return

        try:
            st.sidebar.info("Extracting video stream URL...")
            stream_url = get_youtube_stream_url(source_youtube)
            fps_tracker = FpsTracker()
            st.sidebar.info("Opening video stream...")
            vid_cap = cv2.VideoCapture(stream_url)

            if not vid_cap.isOpened():
                st.sidebar.error(
                    "Failed to open video stream. Please try a different video.")
                return

            st.sidebar.success("Video stream opened successfully!")
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        is_display_tracker,
                        tracker,
                        fps_tracker
                    )
                else:
                    break

            vid_cap.release()

        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")


def play_rtsp_stream(conf, model, settings):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption(
        'Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            fps_tracker = FpsTracker()
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             fps_tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model, settings):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            fps_tracker = FpsTracker()
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             fps_tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model, settings):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    print(f"Using model {model}")
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()
    fps_tracker = FpsTracker()
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)),
                cv2.CAP_FFMPEG)
            st_frame = st.empty()
            #vid_cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.CAP_MSMF )
            #vid_cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.CAP_VAAPI )
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             fps_tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
