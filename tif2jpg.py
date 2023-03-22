import argparse
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tif_dir", help="Directory containing tif files")
    parser.add_argument("jpg_dir", help="Directory to save jpg files")
    args = parser.parse_args()

    # Get all tif files in the directory
    tif_files = Path(args.tif_dir).glob('*.{}'.format("tif"))

    # Convert each tif file to jpg
    for tif_file in tqdm(tif_files):
        # Read the tif file
        img = Image.open(str(tif_file))

        # Get the filename
        filename = tif_file.stem

        # Save the jpg file
        img.save(os.path.join(args.jpg_dir, filename + '.jpg'))
