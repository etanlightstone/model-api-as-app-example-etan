"""Domino Model API entrypoint for an image classifier (the *image* path).

Domino Model APIs take images as a JSON field — typically a **base64-encoded
string** — not multipart. This example follows that contract exactly: the
``image`` argument is a base64 string, which the function decodes, analyzes, and
returns class probabilities for. The Model-API-as-App harness marks the field
as an image (via ``model_app.yaml``) so the playground renders a file picker
that base64-encodes client-side and POSTs the identical JSON endpoint.

The "model" here is intentionally dependency-light (PIL + numpy, no model
download) so it runs anywhere: it classifies an image by simple global pixel
statistics into a handful of visual classes. **Swapping in a real classifier**
(e.g. a torchvision ``resnet50(weights=...)`` or a HF ``ViT`` pipeline) is a
change to the body of ``_classify`` only — the base64 transport, the schema
flag, and the UI file picker are all unchanged.

    File:     example/image_classifier/model_api.py
    Function: predict

Example request body::

    {"image": "<base64-encoded PNG/JPEG bytes>"}

Example response::

    {"label": "colorful", "probabilities": {"dark": 0.1, "bright": 0.2,
     "grayscale": 0.1, "colorful": 0.6}}
"""

from __future__ import annotations

import argparse
import base64
import io

import numpy as np
from PIL import Image

CLASSES = ["dark", "bright", "grayscale", "colorful"]


def _decode(image_b64: str) -> Image.Image:
    """Decode a base64 string (with or without a data-URL prefix) to an image."""
    if "," in image_b64 and image_b64.strip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _classify(img: Image.Image) -> np.ndarray:
    """Return logits over CLASSES from global pixel statistics.

    A stand-in for a real CNN/ViT forward pass — same input/output contract.
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H×W×3
    brightness = arr.mean()
    # Saturation proxy: how far channels spread from the per-pixel mean.
    chan_spread = arr.std(axis=2).mean()
    logits = np.array([
        (0.5 - brightness) * 6.0,             # dark
        (brightness - 0.5) * 6.0,             # bright
        (0.15 - chan_spread) * 12.0,          # grayscale
        (chan_spread - 0.15) * 12.0,          # colorful
    ], dtype=np.float32)
    return logits


def predict(image: str) -> dict:
    """Classify a base64-encoded image into one of CLASSES.

    Domino passes the request's ``image`` JSON field in as this keyword arg.
    """
    img = _decode(image)
    probs = _softmax(_classify(img))
    ranked = {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}
    label = max(ranked, key=ranked.get)
    return {"label": label, "probabilities": ranked}


def main() -> None:
    """Classify an image from the terminal, or a synthetic demo image.

    Reads the file, base64-encodes it (the same transport Domino uses), and
    prints the classification — so the same ``predict`` function runs locally and
    as a hosted Model API.
    """
    p = argparse.ArgumentParser(
        description="Classify an image by global pixel statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", default=None,
                   help="Path to a PNG/JPEG image to classify.")
    args = p.parse_args()

    if args.image:
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    else:
        # No image given -- classify a synthetic bright-blue image.
        print("No --image provided -- classifying a synthetic blue image.")
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (40, 90, 230)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

    print(predict(b64))


if __name__ == "__main__":
    main()
