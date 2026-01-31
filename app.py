# app.py
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image
import streamlit as st

# Import TensorFlow inside cached loader to reduce reruns and keep startup cleaner
# (still fine locally and on Streamlit Cloud)


# ---------- Paths (robust for local + Streamlit Cloud) ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "brain_tumour_model.keras"

# Prefer a TXT if you have it, otherwise fall back to your uploaded JSON
CLASS_NAMES_TXT = BASE_DIR / "class_names.txt"
CLASS_NAMES_JSON = BASE_DIR / "brain_tumour_class_names.json"

IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Brain Tumour MRI Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Lightweight, mobile-friendly styling ----
st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 980px; }
.stButton button { width: 100%; padding: 0.8rem 1rem; border-radius: 14px; }
.card {
  padding: 1rem 1.2rem;
  border-radius: 16px;
  border: 1px solid rgba(120,120,120,0.25);
  background: rgba(250,250,250,0.65);
}
.small { opacity: 0.8; font-size: 0.95rem; }
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  h1 { font-size: 1.6rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def _load_class_names() -> list[str]:
    if CLASS_NAMES_TXT.exists():
        return [
            line.strip()
            for line in CLASS_NAMES_TXT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if CLASS_NAMES_JSON.exists():
        data = json.loads(CLASS_NAMES_JSON.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
        raise ValueError("brain_tumour_class_names.json must be a JSON list of strings.")

    raise FileNotFoundError(
        "Missing class names file. Add class_names.txt or brain_tumour_class_names.json next to app.py"
    )


@st.cache_resource
def load_model_and_classes():
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found: brain_tumour_model.keras (must be next to app.py)"
        )

    model = keras.models.load_model(MODEL_PATH)
    class_names = _load_class_names()
    return model, class_names


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    import tensorflow as tf

    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x


# ---- Header ----
st.title("Brain Tumour MRI Classifier")
st.write("Upload an MRI image to get a prediction and confidence.")

# ---- Options ----
with st.expander("Options", expanded=False):
    show_probs = st.checkbox("Show class probabilities", value=True)
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.01)
    show_topk = st.checkbox("Show top 2 predictions", value=True)

# ---- Load assets with clear errors ----
try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error(f"App failed to start: {e}")
    st.stop()

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Avoid newer Streamlit args that may break on older deployments
col1, col2 = st.columns([1, 1])

with col1:
    if uploaded is None:
        st.markdown(
            """
            <div class="card">
              <b>How to use</b>
              <p class="small">1) Upload a JPG/PNG MRI image<br/>
              2) The model predicts a tumour class with confidence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pil_img = None
    else:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

with col2:
    if pil_img is not None:
        x = preprocess_image(pil_img)

        probs = model.predict(x, verbose=0)[0].astype(float)
        if probs.ndim != 1:
            probs = np.ravel(probs)

        top_idx = int(np.argmax(probs))
        top_label = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"
        top_conf = float(probs[top_idx])

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")

        if top_conf >= threshold:
            st.success(f"{top_label} (confidence: {top_conf:.3f})")
        else:
            st.warning(f"Uncertain: best guess is {top_label} (confidence: {top_conf:.3f})")

        if show_topk:
            top2 = np.argsort(probs)[::-1][:2]
            lines = []
            for i in top2:
                label = class_names[i] if i < len(class_names) else f"class_{i}"
                lines.append(f"- {label}: {float(probs[i]):.3f}")
            st.markdown("**Top predictions**\n" + "\n".join(lines))

        st.markdown(
            '<p class="small">If confidence is low, try a clearer slice or a different image.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if show_probs:
            st.subheader("Class probabilities")
            order = np.argsort(probs)[::-1]
            prob_table = {
                (class_names[i] if i < len(class_names) else f"class_{i}"): float(probs[i])
                for i in order
            }
            # Avoid horizontal=True for compatibility with older Streamlit versions
            st.bar_chart(prob_table)
