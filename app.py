# Python In-built packages
from pathlib import Path
import PIL
import openvino as ov
import openvino.properties as props
import torch

# External packages
import streamlit as st

# Local Modules
from settings import Settings
import helper

# Hiding/workaround of a warning/error? as discussed in https://github.com/VikParuchuri/marker/issues/442
torch.classes.__path__ = []

settings = Settings('yolov8n', 'yolov8n-seg')

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLO v8/11",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection And Tracking using YOLO v8/11")

# Sidebar
st.sidebar.header("ML Model Config")

model_selection = st.sidebar.selectbox(
        "Choose a model...", settings.MODEL_NAMES)

print(f"Current model selected is {model_selection}")
settings.update_model_names(model_selection)
settings.print_model_names()

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Inference engine option
ie_engine_name = st.sidebar.radio(
    "Select inference engine", ['OpenVINO', 'Ultralytics']
    )

core = ov.Core()
devices = core.available_devices

dev_names = []
for dev in devices:
    print ("++++++++++++++++++++++++++")
    dev_names.append(core.get_property(dev, props.device.full_name))
    supported_properties = core.get_property(dev, props.supported_properties)
    indent = len(max(supported_properties, key=len))
    for property_key in supported_properties:
        if property_key not in (
            "SUPPORTED_METRICS",
            "SUPPORTED_CONFIG_KEYS",
            "SUPPORTED_PROPERTIES",
        ):
            try:
                property_val = core.get_property(dev, property_key)
            except TypeError:
                property_val = "UNSUPPORTED TYPE"
            print(f"{property_key:<{indent}}: {property_val}")

print("-------------------------------")
print(dev_names)
ov_device = st.sidebar.radio(
    "Select OpenVINO inference device", core.available_devices, captions=dev_names
    )

ul_model_path=None
# Selecting Detection Or Segmentation
if model_type == 'Detection':
    ul_model_path = Path(settings.DETECTION_MODEL)
    if ie_engine_name == 'OpenVINO':
        model_path = Path(settings.DETECTION_MODEL_OV)
    else:
        model_path = ul_model_path
elif model_type == 'Segmentation':
    ul_model_path = Path(settings.SEGMENTATION_MODEL)
    if ie_engine_name == 'OpenVINO':
        model_path = Path(settings.SEGMENTATION_MODEL_OV)
    else:
        model_path = ul_model_path

# Load Pre-trained ML Model
try:
    if ie_engine_name == 'OpenVINO':
        if model_type == 'Segmentation':
            model = helper.load_model_ov(model_path, ov_device, "segment")
        else:
            model = helper.load_model_ov(model_path, ov_device, "detect")
    elif ie_engine_name == 'Ultralytics':
        model = helper.load_model(model_path)

except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_container_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_container_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence,
                                    device="GPU.1"
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_container_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model, settings)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model, settings)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model, settings)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model, settings)

else:
    st.error("Please select a valid source type!")
