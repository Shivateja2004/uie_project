import os
import cv2
import argparse

from paper_method import enhance_paper_method
from improved_method import enhance as enhance_improved




EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif")

def process_all(dataset, paper_out, improved_out):

    os.makedirs(paper_out, exist_ok=True)
    os.makedirs(improved_out, exist_ok=True)

    files = sorted([f for f in os.listdir(dataset) if f.lower().endswith(EXTS)])

    print(f"Found {len(files)} images in {dataset}")

    for name in files:
        p = os.path.join(dataset, name)
        img = cv2.imread(p)

        if img is None:
            print("Skipping:", name)
            continue

        # PAPER METHOD
        out1 = enhance_paper_method(img)
        cv2.imwrite(os.path.join(paper_out, name), out1)

        # IMPROVED METHOD
        out2 = enhance_improved(img)
        cv2.imwrite(os.path.join(improved_out, name), out2)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="input")
    parser.add_argument("--paper_out", default="paper_output")
    parser.add_argument("--improved_out", default="improved_output")
    args = parser.parse_args()

    process_all(args.dataset, args.paper_out, args.improved_out)
