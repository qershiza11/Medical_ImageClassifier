import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

def run_all(image_path):
    # Train Stage A if missing
    if not os.path.isfile("stageA_tumour_vs_notumour.keras"):
        print("Stage A model not found. Training Stage A...")
        train_stageA()

    # Train Stage B if missing
    if not os.path.isfile("stageB_tumour_type.keras"):
        print("Stage B model not found. Training Stage B...")
        train_stageB()

    # Predict (requires a real image path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    predict_image(image_path)

IMAGE_PATH = r"C:/Users/Asus/PycharmProjects/BrainTumorMRIProject1/braintumorDS/Testing/glioma/Te-gl_0011.jpg"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2

def load_dataset():
    dataset_dir = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    train_dir = os.path.join(dataset_dir, "Training")
    test_dir  = os.path.join(dataset_dir, "Testing")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    return train_ds, val_ds, test_ds

def build_backbone(num_outputs, activation):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_outputs, activation=activation)(x)

    return keras.Model(inputs, outputs), base

def train_stageA():
    train_ds, val_ds, test_ds = load_dataset()
    class_names = train_ds.class_names
    notumor_idx = class_names.index("notumor")

    def to_binary(x, y):
        return x, tf.cast(y != notumor_idx, tf.int32)

    train_ds = train_ds.map(to_binary)
    val_ds   = val_ds.map(to_binary)
    test_ds  = test_ds.map(to_binary)

    model, _ = build_backbone(1, "sigmoid")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", keras.metrics.AUC()]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=8)
    model.evaluate(test_ds)
    model.save("stageA_tumour_vs_notumour.keras")

def train_stageB():
    train_ds, val_ds, test_ds = load_dataset()
    class_names = train_ds.class_names
    tumour_classes = ["glioma", "meningioma", "pituitary"]
    tumour_idx = [class_names.index(c) for c in tumour_classes]

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(tumour_idx),
            tf.constant([0,1,2])
        ),
        default_value=-1
    )

    def filter_remap(ds):
        ds = ds.unbatch()
        ds = ds.filter(lambda x,y: tf.reduce_any(tf.equal(y, tumour_idx)))
        ds = ds.map(lambda x,y: (x, table.lookup(y)))
        return ds.batch(BATCH_SIZE)

    train_ds = filter_remap(train_ds)
    val_ds   = filter_remap(val_ds)
    test_ds  = filter_remap(test_ds)

    model, base = build_backbone(3, "softmax")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=5)

    base.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    model.save("stageB_tumour_type.keras")

def predict_image(image_path):
    stageA = keras.models.load_model("stageA_tumour_vs_notumour.keras")
    stageB = keras.models.load_model("stageB_tumour_type.keras")

    tumour_classes = ["glioma", "meningioma", "pituitary"]

    img = keras.utils.load_img(IMAGE_PATH, target_size=IMG_SIZE)
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    p_tumour = float(stageA.predict(x)[0][0])
    if p_tumour < 0.5:
        print("Prediction: notumor")
        return

    probs = stageB.predict(x)[0]
    idx = int(np.argmax(probs))
    print(f"Prediction: {tumour_classes[idx]} (p_tumour={p_tumour:.3f}, p_type={probs[idx]:.3f})")

def main():
    run_all(IMAGE_PATH)

if __name__ == "__main__":
    main()

