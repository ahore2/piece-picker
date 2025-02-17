import argparse

def main():
    parser = argparse.ArgumentParser(description="Piece Picker")
    parser.add_argument("--image", type=str, help="Image to detect tiny objects from")
    args = parser.parse_args()

if __name__ == "__main__":
    main()