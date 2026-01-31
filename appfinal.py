# appfinal_fixed.py
# Unified 3-model Streamlit app (single folder)
#
# Put these files in the SAME folder as this app:
#   brain_tumour_model.keras
#   pneumonia_cnn_weighted.keras
#   brain_stroke_ct_efficientnet.keras
#   brain_tumour_class_names.json   (or brain_tumor_class_names.json)
#   pneumonia_class_names.json
#   class_names.json
#
# Run:
#   streamlit run appfinal_fixed.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


APP_DIR = Path(__file__).resolve().parent


# -----------------------------
# Model config
# -----------------------------
@dataclass(frozen=True)
class ModelCfg:
    title: str
    model_file: str
    class_names_file: str
    preprocess: str  # "efficientnet" | "mobilenet_v2" | "scale01"


MODEL_INFO: dict[str, ModelCfg] = {
    "Brain tumour MRI": ModelCfg(
        title="Brain tumour MRI",
        model_file="brain_tumour_model.keras",
        class_names_file="brain_tumour_class_names.json",  # will auto-fallback if missing
        preprocess="efficientnet",
    ),
    "Pneumonia X-ray": ModelCfg(
        title="Pneumonia X-ray",
        model_file="pneumonia_cnn_weighted.keras",
        class_names_file="pneumonia_class_names.json",
        preprocess="mobilenet_v2",
    ),
    "Brain stroke CT": ModelCfg(
        title="Brain stroke CT",
        model_file="brain_stroke_ct_efficientnet.keras",
        class_names_file="class_names.json",
        # IMPORTANT:
        # Your stroke model includes EfficientNet preprocessing INSIDE the model graph
        # (x*255 then efficientnet.preprocess_input). If we apply it here again, we
        # double-preprocess and predictions collapse.
        # Therefore we only scale to 0..1 in the app:
        preprocess="scale01",
    ),
}


# -----------------------------
# Helpers
# -----------------------------
def resolve_file_with_fallback(primary: str, fallbacks: list[str]) -> Path:
    """
    Resolve a file in APP_DIR. If primary not found, try fallbacks.
    """
    p = (APP_DIR / primary).resolve()
    if p.exists():
        return p
    for fb in fallbacks:
        q = (APP_DIR / fb).resolve()
        if q.exists():
            return q
    return p  # return primary path even if missing (caller will error nicely)


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    p = (APP_DIR / model_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return tf.keras.models.load_model(p)


@st.cache_data
def load_class_names(class_path: str) -> list[str]:
    p = (APP_DIR / class_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Class names file not found: {p}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError(f"{p.name} must be a JSON list of strings.")
        return data

    if p.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln]

    raise ValueError(f"Unsupported class file format: {p.suffix}. Use .json or .txt")


def clamp01(v: float) -> float:
    return float(min(max(v, 0.0), 1.0))


def get_preprocess_fn(mode: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function that transforms an (H,W,3) float32 array.
    """
    if mode == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input
    if mode == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if mode == "scale01":
        return lambda x: x / 255.0
    raise ValueError(f"Unknown preprocess mode: {mode}")


def preprocess_image(pil_img: Image.Image, model: tf.keras.Model, mode: str) -> np.ndarray:
    """
    Preprocess to match model input shape and expected scaling.
    Supports channel counts 1 and 3.
    """
    in_shape = model.input_shape  # (None,H,W,C)
    if not isinstance(in_shape, (list, tuple)) or len(in_shape) != 4:
        raise ValueError(f"Unexpected model.input_shape: {in_shape}")

    _, H, W, C = in_shape
    if H is None or W is None or C is None:
        raise ValueError(f"Model input shape has None dims: {in_shape}")

    if C == 1:
        img = pil_img.convert("L").resize((W, H))
        x = np.array(img, dtype=np.float32) / 255.0
        x = x[..., None]
        return x[None, ...]

    if C == 3:
        img = pil_img.convert("RGB").resize((W, H))
        x = np.array(img, dtype=np.float32)
        x = get_preprocess_fn(mode)(x)
        return x[None, ...]

    raise ValueError(f"Unsupported channel count C={C} for input_shape={in_shape}")


def predict_probs(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    y = model.predict(x, verbose=0)
    return np.array(y).squeeze()


def is_binary_output(probs: np.ndarray) -> bool:
    # scalar or single-element vector
    if probs.ndim == 0:
        return True
    return probs.ndim == 1 and probs.size == 1


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Medical Image Classifier", layout="wide")
st.title("Medical Image Classifier")

with st.sidebar:
    st.header("Choose model")
    selected_model_name = st.radio("Model", list(MODEL_INFO.keys()), index=0)
    show_debug = st.checkbox("Show debug info", value=True)
    st.caption("Only the selected model is loaded and run.")

cfg = MODEL_INFO[selected_model_name]

# ---- Resolve class file with common spelling fallback
fallbacks = []
if "tumour" in cfg.class_names_file:
    fallbacks.append(cfg.class_names_file.replace("tumour", "tumor"))
if "tumor" in cfg.class_names_file:
    fallbacks.append(cfg.class_names_file.replace("tumor", "tumour"))

classes_path = resolve_file_with_fallback(cfg.class_names_file, fallbacks)
model_path = (APP_DIR / cfg.model_file).resolve()

uploaded = st.file_uploader(
    "Upload an image (MRI / X-ray / CT)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if not uploaded:
    st.info("Pick a model on the left, then upload an image.")
    st.stop()

pil_img = Image.open(uploaded)

left, right = st.columns([1, 1], gap="large")
with left:
    st.image(pil_img, caption="Uploaded image", use_container_width=True)

with right:
    try:
        if show_debug:
            with st.expander("Debug info", expanded=False):
                st.write("Running file:", str(Path(__file__).resolve()))
                st.write("APP_DIR:", str(APP_DIR.resolve()))
                st.write("Selected model:", selected_model_name)
                st.write("Model path:", str(model_path))
                st.write("Classes path:", str(classes_path))
                st.write("Model exists?:", model_path.exists())
                st.write("Classes exists?:", classes_path.exists())
                st.write("Preprocess mode:", cfg.preprocess)
                st.write("Files in APP_DIR:", sorted([p.name for p in APP_DIR.iterdir()]))

        # ---- Load model + class names
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path.name}")
        if not classes_path.exists():
            raise FileNotFoundError(
                f"Missing class file: {Path(cfg.class_names_file).name} "
                f"(also tried: {fallbacks})"
            )

        model = load_model(cfg.model_file)
        class_names = load_class_names(str(classes_path.name))

        # ---- Preprocess + predict
        x = preprocess_image(pil_img, model, cfg.preprocess)
        probs = predict_probs(model, x)

        st.subheader(f"Result: {cfg.title}")

        # ---- Binary
        if is_binary_output(probs):
            p = float(probs) if probs.ndim == 0 else float(probs[0])
            p = clamp01(p)
            st.write(f"Probability (positive class): **{p*100:.2f}%**")
            st.progress(p)

            if isinstance(class_names, list) and len(class_names) == 2:
                st.markdown("**Class probabilities**")
                st.write(f"{class_names[0]}: {(1-p)*100:.2f}%")
                st.write(f"{class_names[1]}: {p*100:.2f}%")

        # ---- Multi-class
        else:
            probs = probs.astype(float).ravel()

            out_dim = model.output_shape[-1] if isinstance(model.output_shape, (list, tuple)) else None
            if isinstance(out_dim, int) and len(class_names) != out_dim:
                raise ValueError(
                    f"Class names length ({len(class_names)}) != model outputs ({out_dim}). "
                    f"Fix {classes_path.name}."
                )

            top_idx = int(np.argmax(probs))
            top_label = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"
            st.write(f"Top prediction: **{top_label}** ({probs[top_idx]*100:.2f}%)")

            st.markdown("**All probabilities**")
            pairs = list(zip(class_names[: len(probs)], probs[: len(class_names)]))
            pairs.sort(key=lambda t: t[1], reverse=True)

            for name, p in pairs:
                p = clamp01(float(p))
                st.write(f"{name}: {p*100:.2f}%")
                st.progress(p)

    except Exception as e:
        st.error("Failed to run the selected model. Check Debug info for file paths and config.")
        st.exception(e)
        st.stop()
