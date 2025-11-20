import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ============================
# PATHS
# ============================
GT_DIR = "ground_truth"
PAPER_DIR = "paper_output"
IMPROVED_DIR = "improved_output"
OUTPUT_CSV = "metrics_results.csv"

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")


# ============================
# --- PCQI ---
# ============================
def PCQI(gt, pr):
    gt = gt.astype(np.float32)
    pr = pr.astype(np.float32)
    eps = 1e-8

    mu_x = cv2.GaussianBlur(gt, (7, 7), 1.5)
    mu_y = cv2.GaussianBlur(pr, (7, 7), 1.5)

    sig_x = np.sqrt(np.abs(cv2.GaussianBlur(gt * gt, (7, 7), 1.5) - mu_x * mu_x))
    sig_y = np.sqrt(np.abs(cv2.GaussianBlur(pr * pr, (7, 7), 1.5) - mu_y * mu_y))

    sig_xy = cv2.GaussianBlur(gt * pr, (7, 7), 1.5) - mu_x * mu_y

    PC = (2 * sig_xy + eps) / (sig_x * sig_x + sig_y * sig_y + eps)
    Q = (2 * mu_x * mu_y + eps) / (mu_x * mu_x + mu_y * mu_y + eps)
    S = (2 * sig_x * sig_y + eps) / (sig_x * sig_x + sig_y * sig_y + eps)

    return float(np.mean(PC * Q * S))


# ============================
# --- UICM ---
# ============================
def UICM(img):
    b, g, r = cv2.split(img.astype(np.float32))

    RG = r - g
    YB = 0.5 * (r + g) - b

    mu_RG = np.mean(RG)
    mu_YB = np.mean(YB)
    sigma_RG = np.std(RG)
    sigma_YB = np.std(YB)

    alpha = 0.3  # from UIQM paper
    return -alpha * (sigma_RG + sigma_YB) + (1 - alpha) * (mu_RG + mu_YB)


# ============================
# --- UISM ---
# ============================
def UISM(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0

    kx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

    ky = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])

    gx = convolve(gray, kx)
    gy = convolve(gray, ky)

    gm = np.sqrt(gx * gx + gy * gy)
    return np.mean(gm)


# ============================
# --- UICONM ---
# ============================
def UICONM(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    return float(np.std(gray))


# ============================
# --- UIQM ---
# ============================
def UIQM(img):
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * UICM(img) + c2 * UISM(img) + c3 * UICONM(img)


# ============================
# --- UCIQE ---
# ============================
def UCIQE(img):
    img = img.astype(np.float32) / 255.0
    lab = rgb2lab(img)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    chroma = np.sqrt(A * A + B * B)
    conl = np.percentile(L, 95) - np.percentile(L, 5)
    ucsd = chroma.std()
    msat = np.mean(chroma)

    return 0.468 * ucsd + 0.2745 * conl + 0.2576 * msat


# ============================
# --- AG ---
# ============================
def AG(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = np.gradient(gray)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


# ============================
# PSNR + SSIM helper
# ============================
def psnr_ssim(gt, pr):
    return (
        psnr(gt, pr, data_range=255),
        ssim(gt, pr, channel_axis=2, data_range=255)
    )


# ============================
# MAIN METRIC LOOP
# ============================
rows = []

paper_tot = np.zeros(8)
improved_tot = np.zeros(8)
count = 0

files = sorted([f for f in os.listdir(GT_DIR) if f.lower().endswith(EXTS)])

for name in files:
    print("Evaluating:", name)

    gt_path = os.path.join(GT_DIR, name)
    paper_path = os.path.join(PAPER_DIR, name)
    imp_path = os.path.join(IMPROVED_DIR, name)

    gt = cv2.imread(gt_path)
    paper = cv2.imread(paper_path)
    imp = cv2.imread(imp_path)

    if gt is None or paper is None or imp is None:
        print("Skipping:", name)
        continue

    # resize mismatch
    if paper.shape != gt.shape:
        paper = cv2.resize(paper, (gt.shape[1], gt.shape[0]))

    if imp.shape != gt.shape:
        imp = cv2.resize(imp, (gt.shape[1], gt.shape[0]))

    # Convert to RGB for SSIM/PSNR
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    paper_rgb = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)
    imp_rgb = cv2.cvtColor(imp, cv2.COLOR_BGR2RGB)

    # ---- PSNR/SSIM ----
    p_psnr, p_ssim = psnr_ssim(gt_rgb, paper_rgb)
    i_psnr, i_ssim = psnr_ssim(gt_rgb, imp_rgb)

    # ---- Underwater Metrics ----
    p_uicm = UICM(paper)
    p_uism = UISM(paper)
    p_uiconm = UICONM(paper)
    p_uiqm = UIQM(paper)
    p_uciqe = UCIQE(paper)
    p_ag = AG(paper)
    p_pcqi = PCQI(gt_rgb, paper_rgb)

    i_uicm = UICM(imp)
    i_uism = UISM(imp)
    i_uiconm = UICONM(imp)
    i_uiqm = UIQM(imp)
    i_uciqe = UCIQE(imp)
    i_ag = AG(imp)
    i_pcqi = PCQI(gt_rgb, imp_rgb)

    rows.append([
        name,
        p_psnr, p_ssim, p_uicm, p_uism, p_uiconm, p_uiqm, p_uciqe, p_ag, p_pcqi,
        i_psnr, i_ssim, i_uicm, i_uism, i_uiconm, i_uiqm, i_uciqe, i_ag, i_pcqi
    ])

    paper_tot += np.array([p_psnr, p_ssim, p_uicm, p_uism, p_uiconm, p_uiqm, p_uciqe, p_ag])
    improved_tot += np.array([i_psnr, i_ssim, i_uicm, i_uism, i_uiconm, i_uiqm, i_uciqe, i_ag])

    count += 1


# SAVE CSV
df = pd.DataFrame(rows, columns=[
    "Image",
    "Paper_PSNR", "Paper_SSIM", "Paper_UICM", "Paper_UISM", "Paper_UICONM",
    "Paper_UIQM", "Paper_UCIQE", "Paper_AG", "Paper_PCQI",
    "Improved_PSNR", "Improved_SSIM", "Improved_UICM", "Improved_UISM",
    "Improved_UICONM", "Improved_UIQM", "Improved_UCIQE", "Improved_AG", "Improved_PCQI"
])

df.to_csv(OUTPUT_CSV, index=False)
print("\nSaved metrics to:", OUTPUT_CSV)


# PRINT AVERAGES
if count > 0:
    print("\n================== AVERAGE METRICS ==================\n")

    paper_avg = paper_tot / count
    imp_avg = improved_tot / count

    print(">>> PAPER METHOD")
    print(f"PSNR     : {paper_avg[0]:.4f}")
    print(f"SSIM     : {paper_avg[1]:.4f}")

    print("\n>>> IMPROVED METHOD")
    print(f"PSNR     : {imp_avg[0]:.4f}")
    print(f"SSIM     : {imp_avg[1]:.4f}")

    print("\n======================================================\n")
