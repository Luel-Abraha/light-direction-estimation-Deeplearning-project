from pathlib import Path
from .model_loader import load_models
from .processing import estimate_light_direction

def main():
    # Load models
    ssis_predictor, depth_processor, depth_model = load_models()

    # Process input image
    input_image_path = Path("inputs/crosswalk.jpg")

    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found at {input_image_path}")

    light_direction = estimate_light_direction(
        input_image_path,
        ssis_predictor,
        depth_processor,
        depth_model, 
        output_dir=Path("outputs")
    )
    print("Estimated light direction (unit vector):", light_direction)

if __name__ == "__main__":
    main()