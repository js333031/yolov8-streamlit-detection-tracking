from pathlib import Path
import requests

from PIL import Image
from ultralytics import YOLO
import nncf
from typing import Dict

int8_model_det_path = Path(f"{DET_MODEL_NAME}_openvino_int8_model/{DET_MODEL_NAME}.xml")
quantized_det_model = None

from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.data.utils import check_det_dataset
from zipfile import ZipFile

from ultralytics.data.utils import DATASETS_DIR

import openvino as ov
from notebook_utils import download_file

core = ov.Core()


DATA_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.1.0/ultralytics/cfg/datasets/coco.yaml"

OUT_DIR = DATASETS_DIR

DATA_PATH = OUT_DIR / "val2017.zip"
LABELS_PATH = OUT_DIR / "coco2017labels-segments.zip"
CFG_PATH = OUT_DIR / "coco.yaml"

# object detection model
det_model_path = Path(f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=True)
    
print("Det model path: ", det_model_path)
det_ov_model = core.read_model(det_model_path)

ov_config = {}
if device.value != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
det_compiled_model = core.compile_model(det_ov_model, device.value, ov_config)
det_model = YOLO(det_model_path.parent, task="detect")

if det_model.predictor is None:
    custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
    args = {**det_model.overrides, **custom}
    det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
    det_model.predictor.setup_model(model=det_model.model)

det_model.predictor.model.ov_compiled_model = det_compiled_model

if not int8_model_det_path.exists():
    if not (OUT_DIR / "coco/labels").exists():
        download_file(DATA_URL, DATA_PATH.name, DATA_PATH.parent)
        download_file(LABELS_URL, LABELS_PATH.name, LABELS_PATH.parent)
        download_file(CFG_URL, CFG_PATH.name, CFG_PATH.parent)
        with ZipFile(LABELS_PATH, "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)
        with ZipFile(DATA_PATH, "r") as zip_ref:
            zip_ref.extractall(OUT_DIR / "coco/images")


    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = str(CFG_PATH)
    det_validator = det_model.task_map[det_model.task]["validator"](args=args)
    det_validator.data = check_det_dataset(args.data)
    det_validator.stride = 32
    det_data_loader = det_validator.get_dataloader(OUT_DIR / "coco", 1)

    det_validator.is_coco = True
    det_validator.class_map = coco80_to_coco91_class()
    det_validator.names = label_map
    det_validator.metrics.names = det_validator.names
    det_validator.nc = 80

if not int8_model_det_path.exists():


    def transform_fn(data_item:Dict):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
        Parameters:
        data_item: Dict with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        input_tensor = det_validator.preprocess(data_item)['img'].numpy()
        return input_tensor


    quantization_dataset = nncf.Dataset(det_data_loader, transform_fn)

if not int8_model_det_path.exists():
    ignored_scope = nncf.IgnoredScope( # post-processing
        subgraphs=[
            nncf.Subgraph(inputs=[f"__module.model.{22 if 'v8' in DET_MODEL_NAME else 23}/aten::cat/Concat",
                                f"__module.model.{22 if 'v8' in DET_MODEL_NAME else 23}/aten::cat/Concat_1",
                                f"__module.model.{22 if 'v8' in DET_MODEL_NAME else 23}/aten::cat/Concat_2"],
                        outputs=[f"__module.model.{22 if 'v8' in DET_MODEL_NAME else 23}/aten::cat/Concat_7"])
        ]
    )

    # Detection model
    quantized_det_model = nncf.quantize(
        det_ov_model,
        quantization_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=ignored_scope
    )
    print(f"Quantized detection model will be saved to {int8_model_det_path}")
    ov.save_model(quantized_det_model, str(int8_model_det_path))


if quantized_det_model is None and int8_model_det_path.exists():
    quantized_det_model = core.read_model(int8_model_det_path)

ov_config = {}
if device.value != "CPU":
    quantized_det_model.reshape({0: [1, 3, 640, 640]})
if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
quantized_det_compiled_model = core.compile_model(quantized_det_model, device.value, ov_config)


if det_model.predictor is None:
    custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
    args = {**det_model.overrides, **custom}
    det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
    det_model.predictor.setup_model(model=det_model.model)

det_model.predictor.model.ov_compiled_model = det_compiled_model

res = det_model(IMAGE_PATH)
display(Image.fromarray(res[0].plot()[:, :, ::-1]))
