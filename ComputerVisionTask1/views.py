import os
import json
import uuid
import numpy as np
import cv2
from PIL import Image
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.conf import settings


# ── Dynamic Kernel Builders ────────────────────────────────────────────────────

def _binomial_row(n):
    row = np.zeros(n + 1, dtype=np.float64)
    row[0] = 1.0
    for i in range(1, n + 1):
        row[i] = row[i - 1] * (n - i + 1) / i
    return row


def make_box_kernel(k):
    return np.ones((k, k), dtype=np.float32) / float(k * k)


def make_gaussian_kernel(k):
    row    = _binomial_row(k - 1).astype(np.float32)
    kernel = np.outer(row, row)
    return kernel / kernel.sum()


def make_sobel_kernels(k):
    half   = k // 2
    smooth = _binomial_row(k - 1).astype(np.float32)
    smooth /= smooth.sum()
    deriv  = np.arange(-half, half + 1, dtype=np.float32)
    return np.outer(smooth, deriv), np.outer(deriv, smooth)


def make_prewitt_kernels(k):
    half   = k // 2
    smooth = np.ones(k, dtype=np.float32) / float(k)
    deriv  = np.arange(-half, half + 1, dtype=np.float32)
    return np.outer(smooth, deriv), np.outer(deriv, smooth)


# ── Convolution ────────────────────────────────────────────────────────────────

def convolve_channel(channel, kernel):
    k   = kernel.shape[0]
    pad = k // 2
    h, w   = channel.shape
    out    = np.zeros((h, w), dtype=np.float32)
    padded = np.pad(channel, pad, mode='edge').astype(np.float32)
    for ky in range(k):
        for kx in range(k):
            out += kernel[ky, kx] * padded[ky:ky + h, kx:kx + w]
    return out


# ── Filter Implementations ────────────────────────────────────────────────────

def _roberts(rgb):
    """
    Roberts Cross edge detector (fixed 2×2 kernels — kernel_size ignored).
    Uses cv2.filter2D for the two cross-diagonal convolutions.
    Returns a grayscale-RGB uint8 array.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Classic 2×2 Roberts kernels
    Rx = np.array([[ 1,  0],
                   [ 0, -1]], dtype=np.float32)
    Ry = np.array([[ 0,  1],
                   [-1,  0]], dtype=np.float32)

    gx  = cv2.filter2D(gray, cv2.CV_32F, Rx)
    gy  = cv2.filter2D(gray, cv2.CV_32F, Ry)
    mag = np.clip(np.sqrt(gx ** 2 + gy ** 2), 0, 255).astype(np.uint8)
    return np.stack([mag, mag, mag], axis=2)


def _canny(rgb, kernel_size=3, low_thresh=50, high_thresh=150):
    """
    Canny edge detector via OpenCV.

    kernel_size  : Gaussian aperture passed to cv2.Canny — must be 3, 5, or 7.
                   Values outside this range are clamped to the nearest valid value.
    low_thresh   : lower hysteresis threshold  (0-255)
    high_thresh  : upper hysteresis threshold  (0-255)
    """
    # cv2.Canny apertureSize must be 3, 5, or 7
    valid_apertures = [3, 5, 7]
    aperture = min(valid_apertures, key=lambda v: abs(v - kernel_size))

    gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=aperture, L2gradient=True)
    return np.stack([edges, edges, edges], axis=2)


def apply_filter(img_array, filter_name, kernel_size=3,
                 canny_low=50, canny_high=150, noise_amount=5):
    """
    Dispatch to the correct filter implementation.

    Parameters
    ----------
    img_array   : H×W×3 uint8 ndarray  (RGB)
    filter_name : 'box' | 'gaussian' | 'sobel' | 'prewitt' | 'roberts' | 'canny'
    kernel_size : odd int ≥ 3  (ignored for roberts; clamped to 3/5/7 for canny)
    canny_low   : Canny lower threshold  (0-255)
    canny_high  : Canny upper threshold  (0-255)
    """
    k   = kernel_size
    rgb = img_array[:, :, :3]

    if filter_name == "box":
        kern   = make_box_kernel(k)
        result = np.stack([
            np.clip(convolve_channel(rgb[:, :, c], kern), 0, 255).astype(np.uint8)
            for c in range(3)
        ], axis=2)

    elif filter_name == "gaussian":
        kern   = make_gaussian_kernel(k)
        result = np.stack([
            np.clip(convolve_channel(rgb[:, :, c], kern), 0, 255).astype(np.uint8)
            for c in range(3)
        ], axis=2)

    elif filter_name in ("sobel", "prewitt"):
        gray = np.mean(rgb, axis=2).astype(np.float32)
        kx_mat, ky_mat = (make_sobel_kernels(k) if filter_name == "sobel"
                          else make_prewitt_kernels(k))
        gx     = convolve_channel(gray, kx_mat)
        gy     = convolve_channel(gray, ky_mat)
        mag    = np.clip(np.sqrt(gx ** 2 + gy ** 2), 0, 255).astype(np.uint8)
        result = np.stack([mag, mag, mag], axis=2)

    elif filter_name == "roberts":
        result = _roberts(rgb)

    elif filter_name == "canny":
        result = _canny(rgb, kernel_size=k,
                        low_thresh=canny_low, high_thresh=canny_high)
        
    elif filter_name == "uniform_noise":
        # Map 100% noise to a maximum range of +/- 127
        intensity = int((noise_amount / 100.0) * 127)
        result = add_uniform_noise(rgb, low=-intensity, high=intensity)
        
    elif filter_name == "gaussian_noise":
        # Map 100% noise to a standard deviation of 127
        std = int((noise_amount / 100.0) * 127)
        result = add_gaussian_noise(rgb, mean=0, std=std)
        
    elif filter_name == "salt_pepper_noise":
        # Map 100% noise to a 1.0 probability (split evenly between salt and pepper)
        prob = noise_amount / 100.0
        result = add_salt_and_pepper_noise(rgb, salt_prob=prob/2, pepper_prob=prob/2)

    else:
        return img_array

    return result


# ── Validation helpers ────────────────────────────────────────────────────────

def parse_kernel_size(raw, default=3):
    """Must be odd integer, 3 ≤ k ≤ 21."""
    MAX_K = 21
    try:
        k = int(raw)
    except (ValueError, TypeError):
        return default, f"Kernel size must be an integer (got '{raw}')."
    if k < 3:
        return default, "Kernel size must be at least 3."
    if k % 2 == 0:
        return default, f"Kernel size must be odd (got {k}). Try {k-1} or {k+1}."
    if k > MAX_K:
        return default, f"Kernel size must be ≤ {MAX_K} (got {k})."
    return k, None


def parse_threshold(raw, default, name):
    """Must be integer 0-255."""
    try:
        v = int(raw)
    except (ValueError, TypeError):
        return default, f"{name} threshold must be an integer (got '{raw}')."
    if not (0 <= v <= 255):
        return default, f"{name} threshold must be between 0 and 255 (got {v})."
    return v, None


# ── Histogram helpers ─────────────────────────────────────────────────────────

def to_grayscale(arr):
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def compute_histogram(gray, L=256):
    hist = [0] * L
    for v in gray.flatten():
        hist[int(v)] += 1
    return hist


def equalize_histogram(gray, L=256):
    n    = gray.size
    hist = compute_histogram(gray, L)
    cdf  = [0] * L
    cdf[0] = hist[0]
    for i in range(1, L):
        cdf[i] = cdf[i - 1] + hist[i]
    lut     = [round((cdf[i] / n) * (L - 1)) for i in range(L)]
    lut_arr = np.array(lut, dtype=np.uint8)
    return lut_arr[gray], lut


# ── File helpers ──────────────────────────────────────────────────────────────

def url_to_filepath(url):
    media_url = settings.MEDIA_URL
    rel = url[len(media_url):] if url.startswith(media_url) else url.lstrip('/')
    return os.path.join(settings.MEDIA_ROOT, rel)


def save_image(pil_img, prefix="out"):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    pil_img.save(os.path.join(settings.MEDIA_ROOT, filename), format="PNG")
    return settings.MEDIA_URL + filename


# ── View ──────────────────────────────────────────────────────────────────────

def home(request):
    # Session defaults
    for key, default in [
        ("img_history",     []),
        ("img_original",    None),
        ("mode",            "filter"),
        ("img_last_filter", ""),
        ("kernel_size",     3),
        ("canny_low",       50),
        ("canny_high",      150),
        ("noise_amount",    5),
    ]:
        if key not in request.session:
            request.session[key] = default

    history  = list(request.session["img_history"])
    original = request.session["img_original"]

    hist_orig_data = request.session.get("hist_orig_data", None)
    hist_eq_data   = request.session.get("hist_eq_data",   None)
    hist_eq_url    = request.session.get("hist_eq_url",    None)

    error = None

    if request.method == "POST":
        action = request.POST.get("action", "")

        # ── Switch mode ───────────────────────────────────────────────────
        if action == "switch_mode":
            request.session["mode"] = request.POST.get("mode", "filter")

        # ── Upload ────────────────────────────────────────────────────────
        elif action == "upload":
            f = request.FILES.get("myfile")
            if f:
                fs         = FileSystemStorage()
                saved_name = fs.save(f.name, f)
                url        = settings.MEDIA_URL + saved_name
                history    = [url]
                original   = url
                request.session["img_last_filter"] = ""
                hist_orig_data = hist_eq_data = hist_eq_url = None

        # ── Apply filter ──────────────────────────────────────────────────
        elif action == "apply_filter":
            filter_name = request.POST.get("filter_name", "").strip()
            raw_k       = request.POST.get("kernel_size",  "3")
            raw_low     = request.POST.get("canny_low",    "50")
            raw_high    = request.POST.get("canny_high",   "150")
            raw_noise   = request.POST.get("noise_amount", "5")

            if not history:
                error = "Please upload an image first."
            elif not filter_name:
                error = "No filter selected."
            else:
                # Validate kernel size (skip for roberts — fixed 2×2)
                if filter_name == "roberts":
                    k, k_error = 3, None      # value unused for roberts
                else:
                    k, k_error = parse_kernel_size(raw_k)

                # Validate Canny thresholds
                canny_low  = request.session["canny_low"]
                canny_high = request.session["canny_high"]
                if filter_name == "canny" and not k_error:
                    canny_low,  e1 = parse_threshold(raw_low,  50,  "Low")
                    canny_high, e2 = parse_threshold(raw_high, 150, "High")
                    if e1:
                        k_error = e1
                    elif e2:
                        k_error = e2
                    elif canny_low >= canny_high:
                        k_error = "Low threshold must be less than high threshold."

                if k_error:
                    error = k_error
                else:
                    # Parse and clamp noise amount between 1 and 100
                    try:
                        noise_amount = max(1, min(100, int(raw_noise)))
                    except ValueError:
                        noise_amount = 5
                        
                    request.session["kernel_size"]  = k
                    request.session["canny_low"]    = canny_low
                    request.session["canny_high"]   = canny_high
                    request.session["noise_amount"] = noise_amount

                    # ── ADD THIS BLOCK TO PREVENT NOISE STACKING ──
                    last_filter = request.session.get("img_last_filter", "")
                    if "noise" in filter_name and "noise" in last_filter:
                        # If applying noise back-to-back, discard the previous noisy 
                        # image so the new noise applies to the clean image before it.
                        if len(history) > 1:
                            history.pop()

                    filepath = url_to_filepath(history[-1])
                    try:
                        pil_img      = Image.open(filepath).convert("RGB")
                        arr          = np.array(pil_img)
                        filtered_arr = apply_filter(
                            arr, filter_name, kernel_size=k,
                            canny_low=canny_low, canny_high=canny_high,
                            noise_amount=noise_amount
                        )
                        filtered_pil = Image.fromarray(filtered_arr)
                        prefix_name  = (f"{filter_name}_{k}x{k}"
                                        if filter_name not in ("roberts", "canny")
                                        else f"{filter_name}")
                        new_url      = save_image(filtered_pil, prefix=prefix_name)
                        history      = history + [new_url]
                        request.session["img_last_filter"] = filter_name
                    except Exception as e:
                        error = f"Filter failed: {e}"

        # ── Undo ──────────────────────────────────────────────────────────
        elif action == "undo":
            if len(history) > 1:
                history = history[:-1]

        # ── Reset ─────────────────────────────────────────────────────────
        elif action == "reset":
            if original:
                history = [original]
                request.session["img_last_filter"] = ""
                request.session["kernel_size"]     = 3
                request.session["canny_low"]       = 50
                request.session["canny_high"]      = 150
                hist_orig_data = hist_eq_data = hist_eq_url = None

        # ── Draw histogram ────────────────────────────────────────────────
        elif action == "draw_histogram":
            if not history:
                error = "Please upload an image first."
            else:
                try:
                    arr            = np.array(Image.open(url_to_filepath(history[0])).convert("RGB"))
                    hist_orig_data = compute_histogram(to_grayscale(arr))
                    hist_eq_data   = hist_eq_url = None
                except Exception as e:
                    error = f"Histogram failed: {e}"

        # ── Equalize ──────────────────────────────────────────────────────
        elif action == "equalize":
            if not history:
                error = "Please upload an image first."
            else:
                try:
                    arr            = np.array(Image.open(url_to_filepath(history[0])).convert("RGB"))
                    gray           = to_grayscale(arr)
                    hist_orig_data = compute_histogram(gray)
                    eq_gray, _     = equalize_histogram(gray)
                    hist_eq_data   = compute_histogram(eq_gray)
                    eq_pil         = Image.fromarray(
                        np.stack([eq_gray, eq_gray, eq_gray], axis=2).astype(np.uint8))
                    hist_eq_url = save_image(eq_pil, prefix="equalized")
                except Exception as e:
                    error = f"Equalization failed: {e}"

        # ── Persist session ───────────────────────────────────────────────
        request.session["img_history"]    = history
        request.session["img_original"]   = original
        request.session["hist_orig_data"] = hist_orig_data
        request.session["hist_eq_data"]   = hist_eq_data
        request.session["hist_eq_url"]    = hist_eq_url
        request.session.modified          = True

    # ── Context ───────────────────────────────────────────────────────────────
    context = {
        "mode":           request.session.get("mode", "filter"),
        "current_url":    history[-1] if history else None,
        "original_url":   original,
        "can_undo":       len(history) > 1,
        "filter_name":    request.session.get("img_last_filter", ""),
        "kernel_size":    request.session.get("kernel_size", 3),
        "canny_low":      request.session.get("canny_low",  50),
        "canny_high":     request.session.get("canny_high", 150),
        "noise_amount":   request.session.get("noise_amount", 5),
        "error":          error,
        "hist_orig_json": json.dumps(hist_orig_data) if hist_orig_data else "null",
        "hist_eq_json":   json.dumps(hist_eq_data)   if hist_eq_data   else "null",
        "hist_eq_url":    hist_eq_url,
        "has_image":      bool(history),
    }
    return render(request, "index.html", context)


# ── Noise Addition ───────────────────────────────────────────────────────────────

# ── Noise Addition ───────────────────────────────────────────────────────────────

def is_effectively_grayscale(img):
    """
    Checks if a 3-channel RGB image is actually grayscale (R == G == B).
    This is needed because our pipeline converts all uploads to RGB.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        # If the Red channel matches Green, and Green matches Blue exactly
        return np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2])
    return False


def add_uniform_noise(img, low=-50, high=50):
    """Adds uniform noise using OpenCV, respecting grayscale vs color."""
    if is_effectively_grayscale(img):
        # Generate 1-channel noise and stack it so it's identical across R, G, B
        noise_1c = np.zeros(img.shape[:2], dtype=np.float32)
        cv2.randu(noise_1c, low, high)
        noise = np.stack([noise_1c, noise_1c, noise_1c], axis=2)
    else:
        # Generate independent 3-channel noise
        noise = np.zeros(img.shape, dtype=np.float32)
        cv2.randu(noise, low, high)
        
    noisy_img = cv2.add(img.astype(np.float32), noise)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_gaussian_noise(img, mean=0, std=25):
    """Adds Gaussian noise using OpenCV, respecting grayscale vs color."""
    if is_effectively_grayscale(img):
        noise_1c = np.zeros(img.shape[:2], dtype=np.float32)
        cv2.randn(noise_1c, mean, std)
        noise = np.stack([noise_1c, noise_1c, noise_1c], axis=2)
    else:
        noise = np.zeros(img.shape, dtype=np.float32)
        cv2.randn(noise, mean, std)
        
    noisy_img = cv2.add(img.astype(np.float32), noise)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02):
    """Adds Salt & Pepper noise using NumPy, respecting grayscale vs color."""
    noisy_img = np.copy(img)
    
    if is_effectively_grayscale(img):
        # Generate 2D probability matrix and duplicate it 3 times
        rand_matrix = np.random.rand(img.shape[0], img.shape[1])
        rand_matrix = np.stack([rand_matrix, rand_matrix, rand_matrix], axis=2)
    else:
        # Generate 3D probability matrix for independent color noise
        rand_matrix = np.random.rand(*img.shape)
        
    noisy_img[rand_matrix < pepper_prob] = 0
    noisy_img[rand_matrix > 1 - salt_prob] = 255
    
    return noisy_img