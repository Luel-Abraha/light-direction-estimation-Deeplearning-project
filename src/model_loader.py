import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from adet.config import get_cfg as get_adet_cfg

from .config import SSIS_CONFIG, SSIS_WEIGHTS, DEPTH_MODEL_NAME

def load_models():
    cfg = get_adet_cfg()
    cfg.merge_from_file(str(SSIS_CONFIG))
    cfg.MODEL.WEIGHTS = str(SSIS_WEIGHTS)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize predictor
    ssis_predictor = DefaultPredictor(cfg)

    # Load Depth Anything model
    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME)
    depth_model = depth_model.to(cfg.MODEL.DEVICE)

    return ssis_predictor, depth_processor, depth_model