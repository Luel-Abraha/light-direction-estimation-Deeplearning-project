from pathlib import Path

# Model paths (in home directory)
SSIS_CONFIG = Path.home() / "SSIS/configs/SSIS/MS_R_101_BiFPN_with_offset_class.yaml"
SSIS_WEIGHTS = Path.home() / "SSIS/tools/output/SSIS_MS_R_101_bifpn_with_offset_class/model_ssis_final.pth"
DEPTH_MODEL_NAME = "LiheYoung/depth-anything-large-hf"

# Input/output paths (relative to project directory)
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)