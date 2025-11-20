# ================================
# PAPER METHOD (Original)
# ================================

import cv2
import numpy as np
import math


# ---------- Stage 1: Quadtree ----------
def compute_flatness_score(patch):
    patch_f = patch.astype(np.float32)/255.0
    means = patch_f.mean(axis=(0,1))
    vars_  = patch_f.var(axis=(0,1))
    return float((means - vars_).mean())


def quadtree_optimal_background(img, min_fraction=0.01):
    H,W = img.shape[:2]
    min_pixels = max(1, int(H*W*min_fraction))
    sm = cv2.bilateralFilter(img,9,75,75)

    stack = [(0,0,W,H)]
    best_score = -1e9
    best_box = (0,0,W,H)

    while stack:
        x0,y0,x1,y1 = stack.pop()
        w,h = x1-x0, y1-y0

        if w*h < min_pixels:
            block = sm[y0:y1, x0:x1]
            if block.size == 0: continue
            s = compute_flatness_score(block)
            if s > best_score:
                best_score = s
                best_box = (x0,y0,x1,y1)
            continue

        mx,my = x0+w//2, y0+h//2
        stack.extend([
            (x0,y0,mx,my),(mx,y0,x1,my),
            (x0,my,mx,y1),(mx,my,x1,y1)
        ])

    return best_box


def get_background_patch_stats(img, box):
    x0,y0,x1,y1 = box
    patch = img[y0:y1, x0:x1].astype(np.float32)/255.0
    return patch.mean(axis=(0,1))


# ---------- Stage 2 & 3: Color Compensation ----------
def classify_color_cast(bg):
    b,g,r = bg
    M = max(bg) - min(bg)
    return (M > 0.2), np.argmax([b,g,r])


def stretch(ch):
    lo, hi = np.percentile(ch,0.1), np.percentile(ch,99.9)
    return np.clip((ch-lo)/(hi-lo+1e-8),0,1)


def adaptive_color_compensation(img, bg):
    imgf = img.astype(np.float32)/255.0
    B,G,R = imgf[...,0], imgf[...,1], imgf[...,2]
    Ib,Ig,Ir = bg

    distorted, idx = classify_color_cast(bg)

    if not distorted:
        return img

    if idx == 2:  # reddish
        Gc = G + 0.7*(Ir-Ig)*(1-G)*R/(R+1e-8)
        Bc = B + 0.7*(Ir-Ib)*(1-B)*R/(R+1e-8)
        Rc = R
    elif idx == 1: # greenish
        Rc = R + 0.7*(Ig-Ir)*(1-R)*G
        Bc = B + 0.7*(Ig-Ib)*(1-B)*G
        Gc = G
    else: # blueish
        Gc = G + 0.7*(Ib-Ig)*(1-G)*B/(B+1e-8)
        Rc = R + 0.7*(Ib-Ir)*(1-R)*G
        Bc = B

    out = np.stack([stretch(Bc), stretch(Gc), stretch(Rc)],axis=-1)
    return (out*255).astype(np.uint8)


# ---------- Stage 4: Homomorphic Filtering ----------
def homomorphic_filter(img, k_override=None):
    imgf = img.astype(np.float32)/255.0

    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)/255.0
    Y = ycc[...,0]

    rows, cols = Y.shape
    D0 = 0.5*np.sqrt(rows**2 + cols**2)

    Ylog = np.log1p(Y)
    F = np.fft.fftshift(np.fft.fft2(Ylog))

    u = np.arange(-cols//2, cols//2)
    v = np.arange(-rows//2, rows//2)
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)+1e-8

    k = 0.8 if k_override is None else k_override
    H = 1 / (1 + np.exp(-(k*(D0/D))))

    G = np.fft.ifft2(np.fft.ifftshift(F * H))
    Y2 = np.expm1(np.real(G))
    Y2 = np.clip(Y2, 0, 1)

    ycc[...,0] = Y2
    return cv2.cvtColor((ycc*255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


# ---------- Stage 5: LAB Fusion ----------
def lab_fusion(color_fixed, hf):
    cc_lab = cv2.cvtColor(color_fixed, cv2.COLOR_BGR2LAB).astype(np.float32)
    hf_lab = cv2.cvtColor(hf, cv2.COLOR_BGR2LAB).astype(np.float32)

    fused = np.stack([hf_lab[...,0], cc_lab[...,1], cc_lab[...,2]],axis=-1)
    return cv2.cvtColor(fused.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------- FINAL PAPER METHOD ----------
def enhance_paper_method(img):
    box = quadtree_optimal_background(img)
    bg = get_background_patch_stats(img, box)
    color_fixed = adaptive_color_compensation(img, bg)
    hf = homomorphic_filter(color_fixed)
    return lab_fusion(color_fixed, hf)
