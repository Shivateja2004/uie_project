import os
import cv2
import numpy as np
import torch
import argparse
from paper_method import enhance_paper_method  # Your main physical/color enhancement

# ---- PyTorch-based Homomorphic Filtering (Fourier Transform) ----
def homomorphic_filter_pytorch(img_bgr, k=0.9):
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)/255.0
    Y = ycc[..., 0]
    H, W = Y.shape
    D0 = 0.5 * np.sqrt(H * H + W * W)
    Ylog = np.log1p(Y)
    Ylog_tensor = torch.from_numpy(Ylog).float()
    F = torch.fft.fftshift(torch.fft.fft2(Ylog_tensor))
    u = torch.arange(-W // 2, W // 2)
    v = torch.arange(-H // 2, H // 2)
    U, V = torch.meshgrid(u, v, indexing="ij")
    D = torch.sqrt(U ** 2 + V ** 2) + 1e-8
    Hf = 1 / (1 + torch.exp(-(k * (D0 / D))))
    F_filtered = F * Hf
    G = torch.fft.ifft2(torch.fft.ifftshift(F_filtered))
    Y2 = np.expm1(G.real.numpy())
    Y2 = np.clip(Y2, 0, 1)
    ycc[..., 0] = Y2
    img_out = cv2.cvtColor((ycc * 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return img_out

# ---- Low-Strength CLAHE (Adaptive Local Contrast) ----
def safe_clahe(img_bgr):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
    img_yuv[...,0] = clahe.apply(img_yuv[...,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# ---- Optional Mild Sharpening (disabled for safety) ----
def mild_sharpen(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (3,3), 0)
    sharp = cv2.addWeighted(img_bgr, 1.03, blur, -0.03, 0)
    return sharp

# ---- Enhancement Pipeline ----
def enhance(img):
    filtered = homomorphic_filter_pytorch(img, k=0.9)     # Step 1: Mild haze/contrast boost
    final = enhance_paper_method(filtered)                 # Step 2: Physical/color enhancement
    final = safe_clahe(final)                             # Step 3: Gentle local contrast
    # Uncomment to test if slight sharpening helps:
    # final = mild_sharpen(final)
    return final

# ---- Batch Folder Processing ----
def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(EXTENSIONS)])
    print(f"Found {len(files)} images in '{input_folder}'.")
    for fname in files:
        fpath = os.path.join(input_folder, fname)
        img = cv2.imread(fpath)
        if img is None:
            print("Skipping:", fname)
            continue
        out = enhance(img)
        cv2.imwrite(os.path.join(output_folder, fname), out)
        print(f"Processed and saved: {fname}")
    print(f"Finished! Enhanced images saved to: '{output_folder}'.")

# ---- Command-Line Interface ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balanced enhancement: PyTorch Fourier filter + paper method + gentle CLAHE")
    parser.add_argument("--input", required=True, help="Input folder with images")
    parser.add_argument("--output", required=True, help="Output folder for enhanced images")
    args = parser.parse_args()
    process_folder(args.input, args.output)
