from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render, get_object_or_404

from .forms import UploadForm
from .models import Upload

import torch
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILES = {
    "fc":  BASE_DIR / "models" / "model_state1_fc.pt",
    "cnn": BASE_DIR / "models" / "model_state2_cnn.pt",
}

_device = torch.device("cpu")
_models = {"fc": None, "cnn": None}


def _load_model(kind: str):
    """kind in {'fc','cnn'}"""
    if _models[kind] is not None:
        return _models[kind]

    if kind == "fc":
        from neural_networks.FullyConnectedClassifier import FullyConnectedClassifier
        m = FullyConnectedClassifier()
        m.load_state_dict(torch.load(MODEL_FILES["fc"], map_location=_device))
    elif kind == "cnn":
        from neural_networks.ConvolutionalClassifier import ConvolutionalClassifier
        m = ConvolutionalClassifier()
        m.load_state_dict(torch.load(MODEL_FILES["cnn"], map_location=_device))
    else:
        raise ValueError("unknown model kind")

    m.eval()
    _models[kind] = m
    return m


def _prep(img_file):
    # grayscale → invert (MNIST-style) → 28×28 → [1,1,28,28] float in [0,1]
    img = Image.open(img_file)
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    x = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return x


@login_required
def upload_view(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            up = form.save(commit=False)
            up.owner = request.user
            up.save()
            return redirect("list")
    else:
        form = UploadForm()
    return render(request, "storageapp/upload.html", {"form": form})


@login_required
def list_view(request):
    items = Upload.objects.filter(owner=request.user).order_by("-created_at")
    return render(request, "storageapp/list.html", {"items": items})


@login_required
def delete_view(request, pk):
    up = get_object_or_404(Upload, pk=pk, owner=request.user)
    up.file.delete(save=False)
    up.delete()
    return redirect("list")


def _classify(up: Upload, kind: str):
    model = _load_model(kind)
    with up.file.open("rb") as f:
        x = _prep(f)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze()
        pred = int(torch.argmax(prob).item())
        p = float(prob[pred].item())
    up.predicted = str(pred)
    up.prob = p
    up.save(update_fields=["predicted", "prob"])
    return pred, p


@login_required
def classify_cnn(request, pk):
    up = get_object_or_404(Upload, pk=pk, owner=request.user)
    pred, prob = _classify(up, kind="cnn")
    up.predicted = str(pred)
    up.prob = prob
    up.model_used = "CNN"       # <---
    up.save(update_fields=["predicted", "prob", "model_used"])
    return redirect("list")


@login_required
def classify_fc(request, pk):
    up = get_object_or_404(Upload, pk=pk, owner=request.user)
    pred, prob = _classify(up, kind="fc")
    up.predicted = str(pred)
    up.prob = prob
    up.model_used = "FullyConnected"  # <---
    up.save(update_fields=["predicted", "prob", "model_used"])
    return redirect("list")
