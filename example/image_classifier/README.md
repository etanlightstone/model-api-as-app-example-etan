# Image classifier example

A deliberately dependency-light image classifier (PIL + numpy, no model
download) that demonstrates the **image** path for both Domino Model APIs and
the [Model-API-as-App harness](../../README.md).

## The contract

Domino Model APIs take images as a JSON field — a **base64-encoded string**, not
multipart. This example follows that exactly:

- **Request:** `{"image": "<base64 PNG/JPEG bytes>"}`
- **Response:** `{"label": "colorful", "probabilities": {"dark": …, "bright": …,
  "grayscale": …, "colorful": …}}`

The `predict(image: str)` function decodes the base64 string, analyzes global
pixel statistics, and returns class probabilities. It's a stand-in for a real
CNN/ViT forward pass — **swapping in a real classifier** (torchvision
`resnet50(weights=…)` or a Hugging Face `ViT` pipeline) is a change to the body
of `_classify` only; the transport, schema, and UI stay the same.

## Hosting it in the app

Point **Settings → Custom function** at:

    File:     example/image_classifier/model_api.py
    Function: predict

The `model_app.yaml` sidecar marks `image` as an image field, so the harness:

- renders a **file picker** in the playground that base64-encodes the file
  client-side and POSTs the identical JSON endpoint, and
- documents the output shape (`label`, `probabilities`) without a probe call.

## Local smoke test

```bash
pip install -r requirements.txt
python model_api.py     # classifies a synthetic blue image
```

## Using a real model

```python
# In model_api.py, replace _classify with, e.g.:
import torch, torchvision as tv
_WEIGHTS = tv.models.ResNet50_Weights.DEFAULT
_MODEL = tv.models.resnet50(weights=_WEIGHTS).eval()
_PRE = _WEIGHTS.transforms()

def _classify_real(img):
    with torch.no_grad():
        logits = _MODEL(_PRE(img).unsqueeze(0))[0]
    return logits.numpy()
```

Add `torch` + `torchvision` to `requirements.txt` and update the `image_fields`
output names in `model_app.yaml` if your class set differs.
