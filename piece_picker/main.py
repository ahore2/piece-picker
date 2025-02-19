import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from piece_picker.objects.model import PiecePicker
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Piece Picker")
    parser.add_argument("--image", type=str, help="Image to detect tiny objects from")
    parser.add_argument("--step_size", type=int, help="Step size in pixels for generating a grid of prompts", default=50)
    parser.add_argument("--upper_threshold", type=float, help="Upper threshold for the ratio of size of the objects and size of the image", default=0.1)
    parser.add_argument("--lower_threshold", type=float, help="Lower threshold for the ratio of size of the objects and size of the image", default=0.01)
    parser.add_argument("--overlap_threshold", type=float, help="Maximum overlap allowed between two objects", default=0.2)
    parser.add_argument("--is_save_inference", type=bool, help="Save the inference image", default=False)
    args = parser.parse_args()

    image = Image.open(args.image)
    image = np.array(image)

    piece_picker = PiecePicker(is_save_inference=args.is_save_inference, 
                               step_size=args.step_size, 
                               upper_threshold=args.upper_threshold, 
                               lower_threshold=args.lower_threshold, 
                               overlap_threshold=args.overlap_threshold)
    piece_picker.predict(image)

if __name__ == "__main__":
    main()