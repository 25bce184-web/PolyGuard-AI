from pathlib import Path
from io import BytesIO
import hashlib
import warnings
from PIL import Image, ImageOps, ImageEnhance
from .config import Settings

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FORMATS = {"JPEG", "PNG", "WEBP"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def load_image(path: Path, settings: Settings) -> Image.Image:
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError("unsupported_extension")
    size = path.stat().st_size
    if size == 0 or size > settings.max_file_mb * 1024 * 1024:
        raise ValueError("empty_or_oversized_file")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as im:
            if im.format not in FORMATS:
                raise ValueError("unsupported_actual_format")
            if im.width * im.height > settings.max_pixels:
                raise ValueError("too_many_pixels")
            im.verify()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            # Composite alpha onto white, not black, then remove EXIF metadata.
            rgba = im.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            return rgb

def preprocess(path: Path, settings: Settings, variant="original", rotation=0):
    im = load_image(path, settings)
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180, or 270")
    if rotation:
        im = im.rotate(-rotation, expand=True)
    im.thumbnail((settings.max_side, settings.max_side), Image.Resampling.LANCZOS)
    if variant == "contrast":
        im = ImageEnhance.Contrast(im).enhance(1.3)
    elif variant == "grayscale":
        im = ImageOps.grayscale(im).convert("RGB")
    elif variant != "original":
        raise ValueError("unknown_variant")
    buffer = BytesIO()
    im.save(buffer, format="PNG")
    if buffer.tell() > 15 * 1024 * 1024:
        raise ValueError("inline_image_too_large_reduce_MAX_IMAGE_SIDE")
    return im, buffer.getvalue()
