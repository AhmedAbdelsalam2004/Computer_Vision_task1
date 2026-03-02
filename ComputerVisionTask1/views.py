import os
import json
import uuid
import base64
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import status


def pil_to_data_url(pil_img):
    """Encode a PIL image as a data URL (PNG)."""
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def data_url_to_pil(value):
    """
    Decode either a data URL or a legacy MEDIA_URL path into a PIL RGB image.
    """
    if not value:
        raise ValueError("Empty image value")
    if value.startswith("data:"):
        try:
            _, b64 = value.split(",", 1)
        except ValueError:
            raise ValueError("Invalid data URL")
        img_bytes = base64.b64decode(b64)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    # Backwards compatibility for existing MEDIA_URL-based paths
    return Image.open(ImageStorage.url_to_filepath(value)).convert("RGB")


# ── Kernel Builders (OOP style) ────────────────────────────────────────────────

class KernelFactory:
    """Builds convolution kernels of various types."""

    @staticmethod
    def _binomial_row(n):
        row = np.zeros(n + 1, dtype=np.float64)
        row[0] = 1.0
        for i in range(1, n + 1):
            row[i] = row[i - 1] * (n - i + 1) / i
        return row

    @staticmethod
    def make_box_kernel(k):
        return np.ones((k, k), dtype=np.float32) / float(k * k)

    @classmethod
    def make_gaussian_kernel(cls, k):
        row = cls._binomial_row(k - 1).astype(np.float32)
        kernel = np.outer(row, row)
        return kernel / kernel.sum()

    @classmethod
    def make_sobel_kernels(cls, k):
        half = k // 2
        smooth = cls._binomial_row(k - 1).astype(np.float32)
        smooth /= smooth.sum()
        deriv = np.arange(-half, half + 1, dtype=np.float32)
        return np.outer(smooth, deriv), np.outer(deriv, smooth)

    @classmethod
    def make_prewitt_kernels(cls, k):
        half = k // 2
        smooth = np.ones(k, dtype=np.float32) / float(k)
        deriv = np.arange(-half, half + 1, dtype=np.float32)
        return np.outer(smooth, deriv), np.outer(deriv, smooth)


# ── Convolution ────────────────────────────────────────────────────────────────

class Convolver:
    """Applies convolution kernels to image channels."""

    @staticmethod
    def convolve_channel(channel, kernel):
        k = kernel.shape[0]
        pad = k // 2
        h, w = channel.shape
        out = np.zeros((h, w), dtype=np.float32)
        padded = np.pad(channel, pad, mode='edge').astype(np.float32)
        for ky in range(k):
            for kx in range(k):
                out += kernel[ky, kx] * padded[ky:ky + h, kx:kx + w]
        return out


# ── Array helpers ─────────────────────────────────────────────────────────────

def normalize_float_to_uint8(arr):
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - mn) * (255.0 / (mx - mn))
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Spatial-domain processors ────────────────────────────────────────────────

class SpatialProcessor:
    """Median filter and normalization (no OpenCV)."""

    @staticmethod
    def median_filter_channel(channel, k):
        pad = k // 2
        padded = np.pad(channel, pad, mode="edge")
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(padded, (k, k))
            med = np.median(windows, axis=(-2, -1))
            return med.astype(np.uint8)
        except Exception:
            # Fallback: pure python loops (slower but correct)
            h, w = channel.shape
            out = np.zeros((h, w), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    block = padded[y:y + k, x:x + k].reshape(-1)
                    out[y, x] = int(np.median(block))
            return out

    @classmethod
    def median_filter_rgb(cls, rgb, k):
        return np.stack([cls.median_filter_channel(rgb[:, :, c], k) for c in range(3)], axis=2)

    @staticmethod
    def normalize_rgb(rgb):
        out = np.zeros_like(rgb, dtype=np.uint8)
        for c in range(3):
            out[:, :, c] = normalize_float_to_uint8(rgb[:, :, c])
        return out


class FrequencyProcessor:
    """Frequency-domain ideal low-pass/high-pass filters (FFT)."""

    @staticmethod
    def _mask(h, w, cutoff_percent, highpass=False):
        cutoff_percent = max(1, min(100, int(cutoff_percent)))
        # radius in pixels (percent of min dimension / 2)
        r = int((cutoff_percent / 100.0) * (min(h, w) / 2.0))
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist2 = (y - cy) ** 2 + (x - cx) ** 2
        mask = (dist2 <= (r * r)).astype(np.float32)
        if highpass:
            mask = 1.0 - mask
        return mask

    @classmethod
    def filter_rgb(cls, rgb, cutoff_percent=30, highpass=False):
        h, w = rgb.shape[:2]
        mask = cls._mask(h, w, cutoff_percent, highpass=highpass)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for c in range(3):
            ch = rgb[:, :, c].astype(np.float32)
            f = np.fft.fft2(ch)
            fshift = np.fft.fftshift(f)
            filtered = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
            mag = np.abs(filtered)
            out[:, :, c] = normalize_float_to_uint8(mag)
        return out


# ── Noise Processor ───────────────────────────────────────────────────────────

class NoiseProcessor:
    """Adds various types of noise to images."""

    @staticmethod
    def is_effectively_grayscale(img):
        """
        Checks if a 3-channel RGB image is actually grayscale (R == G == B).
        This is needed because our pipeline converts all uploads to RGB.
        """
        if len(img.shape) == 3 and img.shape[2] == 3:
            return np.array_equal(img[:, :, 0], img[:, :, 1]) and \
                   np.array_equal(img[:, :, 1], img[:, :, 2])
        return False

    @classmethod
    def add_uniform_noise(cls, img, low=-50, high=50):
        """Adds uniform noise using OpenCV, respecting grayscale vs color."""
        if cls.is_effectively_grayscale(img):
            noise_1c = np.zeros(img.shape[:2], dtype=np.float32)
            cv2.randu(noise_1c, low, high)
            noise = np.stack([noise_1c, noise_1c, noise_1c], axis=2)
        else:
            noise = np.zeros(img.shape, dtype=np.float32)
            cv2.randu(noise, low, high)
        noisy_img = cv2.add(img.astype(np.float32), noise)
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    @classmethod
    def add_gaussian_noise(cls, img, mean=0, std=25):
        """Adds Gaussian noise using OpenCV, respecting grayscale vs color."""
        if cls.is_effectively_grayscale(img):
            noise_1c = np.zeros(img.shape[:2], dtype=np.float32)
            cv2.randn(noise_1c, mean, std)
            noise = np.stack([noise_1c, noise_1c, noise_1c], axis=2)
        else:
            noise = np.zeros(img.shape, dtype=np.float32)
            cv2.randn(noise, mean, std)
        noisy_img = cv2.add(img.astype(np.float32), noise)
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    @classmethod
    def add_salt_and_pepper_noise(cls, img, salt_prob=0.02, pepper_prob=0.02):
        """Adds Salt & Pepper noise using NumPy, respecting grayscale vs color."""
        noisy_img = np.copy(img)
        if cls.is_effectively_grayscale(img):
            rand_matrix = np.random.rand(img.shape[0], img.shape[1])
            rand_matrix = np.stack([rand_matrix, rand_matrix, rand_matrix], axis=2)
        else:
            rand_matrix = np.random.rand(*img.shape)
        noisy_img[rand_matrix < pepper_prob] = 0
        noisy_img[rand_matrix > 1 - salt_prob] = 255
        return noisy_img


# ── Filter Processor ──────────────────────────────────────────────────────────

class FilterProcessor:
    """Applies image filters and edge detectors."""

    @staticmethod
    def _roberts(rgb):
        """
        Roberts Cross edge detector (fixed 2×2 kernels — kernel_size ignored).
        Uses cv2.filter2D for the two cross-diagonal convolutions.
        Returns a grayscale-RGB uint8 array.
        """
        gray = np.mean(rgb.astype(np.float32), axis=2)
        Rx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        Ry = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = Convolver.convolve_channel(gray, Rx)
        gy = Convolver.convolve_channel(gray, Ry)
        mag = normalize_float_to_uint8(np.sqrt(gx ** 2 + gy ** 2))
        return np.stack([mag, mag, mag], axis=2)

    @staticmethod
    def edge_outputs(rgb, filter_name, kernel_size=3):
        """Return (gx_img, gy_img, mag_img) as uint8 grayscale arrays."""
        gray = np.mean(rgb.astype(np.float32), axis=2)

        if filter_name == "roberts":
            Rx = np.array([[1, 0], [0, -1]], dtype=np.float32)
            Ry = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            gx = Convolver.convolve_channel(gray, Rx)
            gy = Convolver.convolve_channel(gray, Ry)
        else:
            kx_mat, ky_mat = (KernelFactory.make_sobel_kernels(kernel_size) if filter_name == "sobel"
                              else KernelFactory.make_prewitt_kernels(kernel_size))
            gx = Convolver.convolve_channel(gray, kx_mat)
            gy = Convolver.convolve_channel(gray, ky_mat)

        gx_img = normalize_float_to_uint8(np.abs(gx))
        gy_img = normalize_float_to_uint8(np.abs(gy))
        mag_img = normalize_float_to_uint8(np.sqrt(gx ** 2 + gy ** 2))
        return gx_img, gy_img, mag_img

    @staticmethod
    def _canny(rgb, kernel_size=3, low_thresh=50, high_thresh=150):
        """
        Canny edge detector via OpenCV.

        kernel_size  : Gaussian aperture passed to cv2.Canny — must be 3, 5, or 7.
                       Values outside this range are clamped to the nearest valid value.
        low_thresh   : lower hysteresis threshold  (0-255)
        high_thresh  : upper hysteresis threshold  (0-255)
        """
        valid_apertures = [3, 5, 7]
        aperture = min(valid_apertures, key=lambda v: abs(v - kernel_size))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=aperture, L2gradient=True)
        return np.stack([edges, edges, edges], axis=2)

    @classmethod
    def apply_filter(cls, img_array, filter_name, kernel_size=3,
                     canny_low=50, canny_high=150, noise_amount=5, freq_cutoff=30):
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
        k = kernel_size
        rgb = img_array[:, :, :3]

        if filter_name == "box":
            kern = KernelFactory.make_box_kernel(k)
            result = np.stack([
                np.clip(Convolver.convolve_channel(rgb[:, :, c], kern), 0, 255).astype(np.uint8)
                for c in range(3)
            ], axis=2)

        elif filter_name == "gaussian":
            kern = KernelFactory.make_gaussian_kernel(k)
            result = np.stack([
                np.clip(Convolver.convolve_channel(rgb[:, :, c], kern), 0, 255).astype(np.uint8)
                for c in range(3)
            ], axis=2)

        elif filter_name in ("sobel", "prewitt"):
            _, _, mag = cls.edge_outputs(rgb, filter_name, kernel_size=k)
            result = np.stack([mag, mag, mag], axis=2)

        elif filter_name == "roberts":
            result = cls._roberts(rgb)

        elif filter_name == "canny":
            result = cls._canny(rgb, kernel_size=k,
                                low_thresh=canny_low, high_thresh=canny_high)

        elif filter_name == "uniform_noise":
            intensity = int((noise_amount / 100.0) * 127)
            result = NoiseProcessor.add_uniform_noise(rgb, low=-intensity, high=intensity)

        elif filter_name == "gaussian_noise":
            std = int((noise_amount / 100.0) * 127)
            result = NoiseProcessor.add_gaussian_noise(rgb, mean=0, std=std)

        elif filter_name == "salt_pepper_noise":
            prob = noise_amount / 100.0
            result = NoiseProcessor.add_salt_and_pepper_noise(rgb, salt_prob=prob / 2, pepper_prob=prob / 2)

        elif filter_name == "median":
            result = SpatialProcessor.median_filter_rgb(rgb, k)

        elif filter_name == "normalize":
            result = SpatialProcessor.normalize_rgb(rgb)

        elif filter_name == "freq_lpf":
            result = FrequencyProcessor.filter_rgb(rgb, cutoff_percent=freq_cutoff, highpass=False)

        elif filter_name == "freq_hpf":
            result = FrequencyProcessor.filter_rgb(rgb, cutoff_percent=freq_cutoff, highpass=True)

        else:
            return img_array

        return result


# ── Histogram Processor ───────────────────────────────────────────────────────

class HistogramProcessor:
    """Computes and equalizes image histograms."""

    @staticmethod
    def to_grayscale(arr):
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    @staticmethod
    def compute_histogram(gray, L=256):
        hist = [0] * L
        for v in gray.flatten():
            hist[int(v)] += 1
        return hist

    @staticmethod
    def compute_cdf(hist):
        return np.cumsum(np.array(hist, dtype=np.int64)).tolist()

    @classmethod
    def compute_rgb_histograms(cls, rgb, L=256):
        r = cls.compute_histogram(rgb[:, :, 0], L)
        g = cls.compute_histogram(rgb[:, :, 1], L)
        b = cls.compute_histogram(rgb[:, :, 2], L)
        return {"r": r, "g": g, "b": b}

    @classmethod
    def equalize_histogram(cls, gray, L=256):
        n = gray.size
        hist = cls.compute_histogram(gray, L)
        cdf = [0] * L
        cdf[0] = hist[0]
        for i in range(1, L):
            cdf[i] = cdf[i - 1] + hist[i]
        lut = [round((cdf[i] / n) * (L - 1)) for i in range(L)]
        lut_arr = np.array(lut, dtype=np.uint8)
        return lut_arr[gray], lut


# ── Validation Helpers ────────────────────────────────────────────────────────

class Validator:
    """Input validation for filter parameters."""

    @staticmethod
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
            return default, f"Kernel size must be odd (got {k}). Try {k - 1} or {k + 1}."
        if k > MAX_K:
            return default, f"Kernel size must be ≤ {MAX_K} (got {k})."
        return k, None

    @staticmethod
    def parse_threshold(raw, default, name):
        """Must be integer 0-255."""
        try:
            v = int(raw)
        except (ValueError, TypeError):
            return default, f"{name} threshold must be an integer (got '{raw}')."
        if not (0 <= v <= 255):
            return default, f"{name} threshold must be between 0 and 255 (got {v})."
        return v, None


# ── File Helpers ──────────────────────────────────────────────────────────────

class ImageStorage:
    """Handles file system operations for images."""

    @staticmethod
    def url_to_filepath(url):
        media_url = settings.MEDIA_URL
        rel = url[len(media_url):] if url.startswith(media_url) else url.lstrip('/')
        return os.path.join(settings.MEDIA_ROOT, rel)

    @staticmethod
    def save_image(pil_img, prefix="out"):
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        pil_img.save(os.path.join(settings.MEDIA_ROOT, filename), format="PNG")
        return settings.MEDIA_URL + filename


# ── Session Manager ───────────────────────────────────────────────────────────

class SessionManager:
    """Manages session state for the image processing pipeline."""

    DEFAULTS = {
        "img_history": [],
        "img_original": None,
        "mode": "filter",
        "img_last_filter": "",
        "filter_history": [],
        "kernel_size": 3,
        "canny_low": 50,
        "canny_high": 150,
        "noise_amount": 5,
        "freq_cutoff": 30,
        "edge_x_url": None,
        "edge_y_url": None,
        "hist_orig_cdf": None,
        "hist_eq_cdf": None,
        "hist_rgb_data": None,
        "hist_rgb_cdf": None,
        # Hybrid image defaults
        "hybrid_low_url": None,
        "hybrid_high_url": None,
        "hybrid_out_url": None,
    }

    @classmethod
    def init_session(cls, session):
        for key, default in cls.DEFAULTS.items():
            if key not in session:
                session[key] = default

    @staticmethod
    def get_state(session):
        return {
            "img_history": list(session.get("img_history", [])),
            "img_original": session.get("img_original"),
            "mode": session.get("mode", "filter"),
            "img_last_filter": session.get("img_last_filter", ""),
            "filter_history": list(session.get("filter_history", [])),
            "kernel_size": session.get("kernel_size", 3),
            "canny_low": session.get("canny_low", 50),
            "canny_high": session.get("canny_high", 150),
            "noise_amount": session.get("noise_amount", 5),
            "freq_cutoff": session.get("freq_cutoff", 30),
            "edge_x_url": session.get("edge_x_url"),
            "edge_y_url": session.get("edge_y_url"),
            "hist_orig_data": session.get("hist_orig_data"),
            "hist_orig_cdf": session.get("hist_orig_cdf"),
            "hist_eq_data": session.get("hist_eq_data"),
            "hist_eq_cdf": session.get("hist_eq_cdf"),
            "hist_eq_url": session.get("hist_eq_url"),
            "hist_rgb_data": session.get("hist_rgb_data"),
            "hist_rgb_cdf": session.get("hist_rgb_cdf"),
            "hybrid_low_url": session.get("hybrid_low_url"),
            "hybrid_high_url": session.get("hybrid_high_url"),
            "hybrid_out_url": session.get("hybrid_out_url"),
        }

    @staticmethod
    def save_state(session, state):
        for key, value in state.items():
            session[key] = value
        session.modified = True


# ── API Views (Class-Based) ───────────────────────────────────────────────────

class StateView(APIView):
    """GET /api/state/ — Returns current session state."""

    def get(self, request):
        SessionManager.init_session(request.session)
        state = SessionManager.get_state(request.session)
        history = state["img_history"]
        return Response({
            "mode": state["mode"],
            "current_url": history[-1] if history else None,
            "original_url": state["img_original"],
            "can_undo": len(history) > 1,
            "has_image": bool(history),
            "filter_name": state["img_last_filter"],
            "kernel_size": state["kernel_size"],
            "canny_low": state["canny_low"],
            "canny_high": state["canny_high"],
            "noise_amount": state["noise_amount"],
            "freq_cutoff": state.get("freq_cutoff", 30),
            "edge_x_url": state.get("edge_x_url"),
            "edge_y_url": state.get("edge_y_url"),
            "hist_orig_data": state["hist_orig_data"],
            "hist_orig_cdf": state.get("hist_orig_cdf"),
            "hist_eq_data": state["hist_eq_data"],
            "hist_eq_cdf": state.get("hist_eq_cdf"),
            "hist_eq_url": state["hist_eq_url"],
            "hist_rgb_data": state.get("hist_rgb_data"),
            "hist_rgb_cdf": state.get("hist_rgb_cdf"),
            "history_length": len(history),
            "hybrid_low_url": state.get("hybrid_low_url"),
            "hybrid_high_url": state.get("hybrid_high_url"),
            "hybrid_out_url": state.get("hybrid_out_url"),
        })


class UploadView(APIView):
    """POST /api/upload/ — Upload an image file."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        SessionManager.init_session(request.session)
        f = request.FILES.get("myfile")
        if not f:
            return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            pil_img = Image.open(f).convert("RGB")
        except Exception as e:
            return Response({"error": f"Invalid image file: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        data_url = pil_to_data_url(pil_img)

        state = SessionManager.get_state(request.session)
        state["img_history"] = [data_url]
        state["img_original"] = data_url
        state["img_last_filter"] = ""
        state["filter_history"] = [""]
        state["edge_x_url"] = None
        state["edge_y_url"] = None
        state["hist_orig_data"] = None
        state["hist_orig_cdf"] = None
        state["hist_eq_data"] = None
        state["hist_eq_cdf"] = None
        state["hist_eq_url"] = None
        state["hist_rgb_data"] = None
        state["hist_rgb_cdf"] = None
        state["hybrid_low_url"] = None
        state["hybrid_high_url"] = None
        state["hybrid_out_url"] = None
        SessionManager.save_state(request.session, state)

        return Response({
            "current_url": data_url,
            "original_url": data_url,
            "can_undo": False,
            "has_image": True,
            "history_length": 1,
            "hist_orig_data": None,
            "hist_eq_data": None,
            "hist_eq_url": None,
            "hist_orig_cdf": None,
            "hist_eq_cdf": None,
            "hist_rgb_data": None,
            "hist_rgb_cdf": None,
            "edge_x_url": None,
            "edge_y_url": None,
        })


class ApplyFilterView(APIView):
    """POST /api/apply-filter/ — Apply a filter to the current image."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def post(self, request):
        SessionManager.init_session(request.session)
        state = SessionManager.get_state(request.session)
        history = state["img_history"]

        filter_name = request.data.get("filter_name", "").strip()
        raw_k = request.data.get("kernel_size", "3")
        raw_low = request.data.get("canny_low", "50")
        raw_high = request.data.get("canny_high", "150")
        raw_noise = request.data.get("noise_amount", "5")
        raw_cutoff = request.data.get("freq_cutoff", state.get("freq_cutoff", 30))

        if not history:
            return Response({"error": "Please upload an image first."}, status=400)
        if not filter_name:
            return Response({"error": "No filter selected."}, status=400)

        # Parse / validate frequency cutoff (1-100)
        try:
            freq_cutoff = max(1, min(100, int(raw_cutoff)))
        except (ValueError, TypeError):
            freq_cutoff = int(state.get("freq_cutoff", 30))

        # Validate kernel size (not needed for normalize/frequency filters)
        if filter_name in ("normalize", "freq_lpf", "freq_hpf"):
            k, k_error = int(state.get("kernel_size", 3)), None
        elif filter_name == "roberts":
            k, k_error = 3, None
        else:
            k, k_error = Validator.parse_kernel_size(raw_k)

        # Validate Canny thresholds
        canny_low = state["canny_low"]
        canny_high = state["canny_high"]
        if filter_name == "canny" and not k_error:
            canny_low, e1 = Validator.parse_threshold(raw_low, 50, "Low")
            canny_high, e2 = Validator.parse_threshold(raw_high, 150, "High")
            if e1:
                k_error = e1
            elif e2:
                k_error = e2
            elif canny_low >= canny_high:
                k_error = "Low threshold must be less than high threshold."

        if k_error:
            return Response({"error": k_error}, status=400)

        # Parse noise amount
        try:
            noise_amount = max(1, min(100, int(raw_noise)))
        except ValueError:
            noise_amount = 5

        # Prevent noise stacking
        last_filter = state.get("img_last_filter", "")
        if "noise" in filter_name and "noise" in last_filter:
            if len(history) > 1:
                history = history[:-1]

        edge_x_url = None
        edge_y_url = None
        try:
            pil_img = data_url_to_pil(history[-1])
            arr = np.array(pil_img)

            if filter_name in ("sobel", "prewitt", "roberts"):
                gx, gy, mag = FilterProcessor.edge_outputs(arr[:, :, :3], filter_name, kernel_size=k)
                mag_rgb = np.stack([mag, mag, mag], axis=2)
                gx_rgb = np.stack([gx, gx, gx], axis=2)
                gy_rgb = np.stack([gy, gy, gy], axis=2)
                edge_x_url = pil_to_data_url(Image.fromarray(gx_rgb))
                edge_y_url = pil_to_data_url(Image.fromarray(gy_rgb))
                new_data_url = pil_to_data_url(Image.fromarray(mag_rgb))
            else:
                filtered_arr = FilterProcessor.apply_filter(
                    arr, filter_name, kernel_size=k,
                    canny_low=canny_low, canny_high=canny_high,
                    noise_amount=noise_amount,
                    freq_cutoff=freq_cutoff,
                )
                filtered_pil = Image.fromarray(filtered_arr)
                new_data_url = pil_to_data_url(filtered_pil)

            history = history + [new_data_url]
        except Exception as e:
            return Response({"error": f"Filter failed: {e}"}, status=500)

        # Update session
        state["img_history"] = history
        state["kernel_size"] = k
        state["canny_low"] = canny_low
        state["canny_high"] = canny_high
        state["noise_amount"] = noise_amount
        state["freq_cutoff"] = freq_cutoff
        state["img_last_filter"] = filter_name
        state["edge_x_url"] = edge_x_url
        state["edge_y_url"] = edge_y_url
        # Clear histogram caches (avoid stale plots after edits)
        state["hist_orig_data"] = None
        state["hist_orig_cdf"] = None
        state["hist_eq_data"] = None
        state["hist_eq_cdf"] = None
        state["hist_eq_url"] = None
        state["hist_rgb_data"] = None
        state["hist_rgb_cdf"] = None

        fh = state.get("filter_history") or []
        if len(fh) > len(history) - 1:
            fh = fh[:len(history) - 1]
        if len(fh) < len(history) - 1:
            fh = fh + [""] * ((len(history) - 1) - len(fh))
        fh.append(filter_name)
        state["filter_history"] = fh
        SessionManager.save_state(request.session, state)

        return Response({
            "current_url": new_data_url,
            "original_url": state["img_original"],
            "can_undo": len(history) > 1,
            "has_image": True,
            "filter_name": filter_name,
            "kernel_size": k,
            "freq_cutoff": freq_cutoff,
            "edge_x_url": edge_x_url,
            "edge_y_url": edge_y_url,
            "history_length": len(history),
        })


class UndoView(APIView):
    """POST /api/undo/ — Undo the last applied filter."""

    def post(self, request):
        SessionManager.init_session(request.session)
        state = SessionManager.get_state(request.session)
        history = state["img_history"]
        fh = state.get("filter_history") or []

        if len(history) > 1:
            history = history[:-1]
            state["img_history"] = history
            if len(fh) > 1:
                fh = fh[:-1]
            state["filter_history"] = fh
            state["img_last_filter"] = fh[-1] if fh else ""
            state["edge_x_url"] = None
            state["edge_y_url"] = None
            SessionManager.save_state(request.session, state)

        return Response({
            "current_url": history[-1] if history else None,
            "original_url": state["img_original"],
            "can_undo": len(history) > 1,
            "has_image": bool(history),
            "filter_name": state.get("img_last_filter", ""),
            "edge_x_url": None,
            "edge_y_url": None,
            "history_length": len(history),
        })


class ResetView(APIView):
    """POST /api/reset/ — Reset to the original uploaded image."""

    def post(self, request):
        SessionManager.init_session(request.session)
        state = SessionManager.get_state(request.session)
        original = state["img_original"]

        if original:
            state["img_history"] = [original]
            state["img_last_filter"] = ""
            state["filter_history"] = [""]
            state["kernel_size"] = 3
            state["canny_low"] = 50
            state["canny_high"] = 150
            state["freq_cutoff"] = 30
            state["edge_x_url"] = None
            state["edge_y_url"] = None
            state["hist_orig_data"] = None
            state["hist_orig_cdf"] = None
            state["hist_eq_data"] = None
            state["hist_eq_cdf"] = None
            state["hist_eq_url"] = None
            state["hist_rgb_data"] = None
            state["hist_rgb_cdf"] = None
            SessionManager.save_state(request.session, state)

        return Response({
            "current_url": original,
            "original_url": original,
            "can_undo": False,
            "has_image": bool(original),
            "hist_orig_data": None,
            "hist_eq_data": None,
            "hist_eq_url": None,
            "hist_orig_cdf": None,
            "hist_eq_cdf": None,
            "hist_rgb_data": None,
            "hist_rgb_cdf": None,
            "filter_name": "",
            "edge_x_url": None,
            "edge_y_url": None,
            "freq_cutoff": 30,
            "history_length": 1 if original else 0,
        })


class DrawHistogramView(APIView):
    """POST /api/histogram/ — Compute histogram of the original image."""

    def post(self, request):
        SessionManager.init_session(request.session)
        state = SessionManager.get_state(request.session)
        history = state["img_history"]

        if not history:
            return Response({"error": "Please upload an image first."}, status=400)

        try:
            arr = np.array(data_url_to_pil(history[0]))
            rgb = arr[:, :, :3].astype(np.uint8)
            gray = HistogramProcessor.to_grayscale(rgb)
            hist_orig_data = HistogramProcessor.compute_histogram(gray)
            hist_orig_cdf = HistogramProcessor.compute_cdf(hist_orig_data)
            hist_rgb_data = HistogramProcessor.compute_rgb_histograms(rgb)
            hist_rgb_cdf = {k: HistogramProcessor.compute_cdf(v) for k, v in hist_rgb_data.items()}
            state["hist_orig_data"] = hist_orig_data
            state["hist_orig_cdf"] = hist_orig_cdf
            state["hist_eq_data"] = None
            state["hist_eq_cdf"] = None
            state["hist_eq_url"] = None
            state["hist_rgb_data"] = hist_rgb_data
            state["hist_rgb_cdf"] = hist_rgb_cdf
            SessionManager.save_state(request.session, state)
            return Response({
                "hist_orig_data": hist_orig_data,
                "hist_orig_cdf": hist_orig_cdf,
                "hist_rgb_data": hist_rgb_data,
                "hist_rgb_cdf": hist_rgb_cdf,
                "hist_eq_data": None,
                "hist_eq_cdf": None,
                "hist_eq_url": None,
            })
        except Exception as e:
            return Response({"error": f"Histogram failed: {e}"}, status=500)


class EqualizeView(APIView):
    """POST /api/equalize/ — Equalize histogram and return both histograms + equalized image URL."""

    def post(self, request):
        SessionManager.init_session(request.session)
        state = SessionManager.get_state(request.session)
        history = state["img_history"]

        if not history:
            return Response({"error": "Please upload an image first."}, status=400)

        try:
            arr = np.array(data_url_to_pil(history[0]))
            gray = HistogramProcessor.to_grayscale(arr)
            hist_orig_data = HistogramProcessor.compute_histogram(gray)
            hist_orig_cdf = HistogramProcessor.compute_cdf(hist_orig_data)
            eq_gray, _ = HistogramProcessor.equalize_histogram(gray)
            hist_eq_data = HistogramProcessor.compute_histogram(eq_gray)
            hist_eq_cdf = HistogramProcessor.compute_cdf(hist_eq_data)
            eq_pil = Image.fromarray(
                np.stack([eq_gray, eq_gray, eq_gray], axis=2).astype(np.uint8)
            )
            hist_eq_url = pil_to_data_url(eq_pil)

            state["hist_orig_data"] = hist_orig_data
            state["hist_orig_cdf"] = hist_orig_cdf
            state["hist_eq_data"] = hist_eq_data
            state["hist_eq_cdf"] = hist_eq_cdf
            state["hist_eq_url"] = hist_eq_url
            state["edge_x_url"] = None
            state["edge_y_url"] = None

            # Append equalized image to temp history for undo/reset
            history = history + [hist_eq_url]
            state["img_history"] = history
            fh = state.get("filter_history") or []
            if len(fh) > len(history) - 1:
                fh = fh[:len(history) - 1]
            if len(fh) < len(history) - 1:
                fh = fh + [""] * ((len(history) - 1) - len(fh))
            fh.append("equalize")
            state["filter_history"] = fh
            state["img_last_filter"] = "equalize"
            SessionManager.save_state(request.session, state)

            return Response({
                "hist_orig_data": hist_orig_data,
                "hist_orig_cdf": hist_orig_cdf,
                "hist_eq_data": hist_eq_data,
                "hist_eq_cdf": hist_eq_cdf,
                "hist_eq_url": hist_eq_url,
                "current_url": hist_eq_url,
                "original_url": state["img_original"],
                "can_undo": len(history) > 1,
                "has_image": True,
                "filter_name": "equalize",
                "edge_x_url": None,
                "edge_y_url": None,
                "history_length": len(history),
            })
        except Exception as e:
            return Response({"error": f"Equalization failed: {e}"}, status=500)


class HybridLowUploadView(APIView):
    """POST /api/hybrid-low/ — Upload first image and apply low-pass (Box, 5×5)."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        SessionManager.init_session(request.session)
        f = request.FILES.get("img_low")
        if not f:
            return Response({"error": "No file provided for low-pass image."}, status=400)

        try:
            pil_img = Image.open(f).convert("RGB")
            arr = np.array(pil_img)
            filtered_arr = FilterProcessor.apply_filter(
                arr, "box", kernel_size=5, canny_low=50, canny_high=150, noise_amount=5
            )
            filtered_pil = Image.fromarray(filtered_arr)
            low_url = pil_to_data_url(filtered_pil)
        except Exception as e:
            return Response({"error": f"Hybrid low-pass failed: {e}"}, status=500)

        state = SessionManager.get_state(request.session)
        state["hybrid_low_url"] = low_url
        # Clear previous mixed output when changing inputs
        state["hybrid_out_url"] = None
        SessionManager.save_state(request.session, state)

        return Response({
            "hybrid_low_url": low_url,
            "hybrid_out_url": None,
        })


class HybridHighUploadView(APIView):
    """POST /api/hybrid-high/ — Upload second image and apply high-pass (Sobel, 5×5)."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        SessionManager.init_session(request.session)
        f = request.FILES.get("img_high")
        if not f:
            return Response({"error": "No file provided for high-pass image."}, status=400)

        try:
            pil_img = Image.open(f).convert("RGB")
            arr = np.array(pil_img)
            filtered_arr = FilterProcessor.apply_filter(
                arr, "sobel", kernel_size=5, canny_low=50, canny_high=150, noise_amount=5
            )
            filtered_pil = Image.fromarray(filtered_arr)
            high_url = pil_to_data_url(filtered_pil)
        except Exception as e:
            return Response({"error": f"Hybrid high-pass failed: {e}"}, status=500)

        state = SessionManager.get_state(request.session)
        state["hybrid_high_url"] = high_url
        # Clear previous mixed output when changing inputs
        state["hybrid_out_url"] = None
        SessionManager.save_state(request.session, state)

        return Response({
            "hybrid_high_url": high_url,
            "hybrid_out_url": None,
        })


class HybridMixView(APIView):
    """POST /api/hybrid-mix/ — Combine low-pass and high-pass images into a hybrid image."""

    def post(self, request):
        SessionManager.init_session(request.session)
        state = SessionManager.get_state(request.session)
        low_url = state.get("hybrid_low_url")
        high_url = state.get("hybrid_high_url")

        if not low_url or not high_url:
            return Response({"error": "Please upload both images before mixing."}, status=400)

        try:
            low_arr = np.array(data_url_to_pil(low_url)).astype(np.float32)
            high_arr = np.array(data_url_to_pil(high_url)).astype(np.float32)

            # Ensure both images have the same size
            if low_arr.shape != high_arr.shape:
                h, w = low_arr.shape[:2]
                high_pil_resized = Image.fromarray(high_arr.astype(np.uint8)).resize((w, h))
                high_arr = np.array(high_pil_resized).astype(np.float32)

            hybrid_arr = np.clip(low_arr + high_arr, 0, 255).astype(np.uint8)
            hybrid_pil = Image.fromarray(hybrid_arr)
            hybrid_url = pil_to_data_url(hybrid_pil)
        except Exception as e:
            return Response({"error": f"Hybrid mix failed: {e}"}, status=500)

        state["hybrid_out_url"] = hybrid_url
        SessionManager.save_state(request.session, state)

        return Response({
            "hybrid_out_url": hybrid_url,
        })


class SwitchModeView(APIView):
    """POST /api/switch-mode/ — Switch between 'filter' and 'histogram' modes."""

    def post(self, request):
        SessionManager.init_session(request.session)
        mode = request.data.get("mode", "filter")
        request.session["mode"] = mode
        request.session.modified = True
        return Response({"mode": mode})