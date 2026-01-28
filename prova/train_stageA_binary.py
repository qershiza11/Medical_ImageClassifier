import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    tf.keras.utils.set_random_seed(42)

    dataset_dir = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    train_dir = os.path.join(dataset_dir, "Training")
    test_dir  = os.path.join(dataset_dir, "Testing")

    img_size = (224, 224)
    batch_size = 32
    val_split = 0.2

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Original classes:", class_names)

    # Map labels to binary: 0=notumor, 1=tumour
    notumor_idx = class_names.index("notumor")

    def to_binary(x, y):
        y_bin = tf.cast(tf.not_equal(y, notumor_idx), tf.int32)
        return x, y_bin

    train_bin = train_ds.map(to_binary)
    val_bin   = val_ds.map(to_binary)
    test_bin  = test_ds.map(to_binary)

    AUTOTUNE = tf.data.AUTOTUNE
    train_bin = train_bin.cache().prefetch(AUTOTUNE)
    val_bin   = val_bin.cache().prefetch(AUTOTUNE)
    test_bin  = test_bin.cache().prefetch(AUTOTUNE)

    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=img_size + (3,)
    )
    base.trainable = False

    inputs = keras.Input(shape=img_size + (3,))
    x = data_aug(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # binary

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("stageA_tumour_vs_notumour.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ]

    model.fit(train_bin, validation_data=val_bin, epochs=8, callbacks=callbacks)
    model.evaluate(test_bin)

    print("Saved: stageA_tumour_vs_notumour.keras")

if __name__ == "__main__":
    main()
