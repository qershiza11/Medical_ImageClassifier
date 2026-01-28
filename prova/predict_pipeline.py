import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (224, 224)

def load_img(path):
    img = keras.utils.load_img(path, target_size=IMG_SIZE)
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x

def main():
    stageA = keras.models.load_model("stageA_tumour_vs_notumour.keras")
    stageB = keras.models.load_model("stageB_tumour_type.keras")

    tumour_classes = ["glioma", "meningioma", "pituitary"]

    image_path = r"C:\Users\Asus\PycharmProjects\BrainTumorMRIProject1\braintumorDS\Testing\glioma\Te-gl_0011.jpg"
    x = load_img(image_path)

    p_tumour = float(stageA.predict(x)[0][0])
    if p_tumour < 0.5:
        print(f"Prediction: notumor (p_tumour={p_tumour:.3f})")
        return

    probs = stageB.predict(x)[0]
    idx = int(np.argmax(probs))
    print(f"Prediction: {tumour_classes[idx]} (p_tumour={p_tumour:.3f}, p_type={probs[idx]:.3f})")

if __name__ == "__main__":
    main()
