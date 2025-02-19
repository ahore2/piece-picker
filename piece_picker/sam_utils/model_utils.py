from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from piece_picker.sam_utils.utils import get_device

def get_sam2_predictor(checkpoint_path = "checkpoint/sam2.1_hiera_tiny.pt", model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml", device = None):

    if not device:
        device = get_device()

    sam2_checkpoint = checkpoint_path
    model_cfg = model_cfg

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)
    return predictor