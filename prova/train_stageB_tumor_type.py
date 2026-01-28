# train_stageB_tumor_type.py
# Stage B: tumour-type classifier (glioma vs meningioma vs pituitary)
# Uses the KaggleHub dataset and filters out "notumor" safely (unbatch -> filter -> remap -> rebatch)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


def main():
    tf.keras.utils.set_random_seed(42)

    # 1) Download dataset
    dataset_dir = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print("Dataset downloaded to:", dataset_dir)

    train_dir = os.path.join(dataset_dir, "Training")
    test_dir = os.path.join(dataset_dir, "Testing")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Expected folders not found.\n"
            f"Looked for:\n  {train_dir}\n  {test_dir}\n"
            f"Check the downloaded folder structure."
        )

    # 2) Hyperparameters
    img_size = (224, 224)
    batch_size = 32
    val_split = 0.2
    epochs_head = 5
    epochs_finetune = 10

    # 3) Load datasets
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

    tumour_classes = ["glioma", "meningioma", "pituitary"]
    tumour_idx = [class_names.index(c) for c in tumour_classes]  # original label ids
    tumour_idx_tf = tf.constant(tumour_idx, dtype=tf.int32)

    # Lookup table: original label -> new label (0..2)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(tumour_idx, dtype=tf.int32),
            values=tf.constant([0, 1, 2], dtype=tf.int32)
        ),
        default_value=-1
    )

    def filter_remap_rebatch(ds):
        # Work per-sample to avoid shape mismatch issues
        ds = ds.unbatch()

        # Keep only tumour samples (drop "notumor")
        ds = ds.filter(lambda x, y: tf.reduce_any(tf.equal(tf.cast(y, tf.int32), tumour_idx_tf)))

        # Remap labels to 0..2
        ds = ds.map(
            lambda x, y: (x, table.lookup(tf.cast(y, tf.int32))),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch back up
        ds = ds.batch(batch_size)
        return ds

    train_t = filter_remap_rebatch(train_ds)
    val_t = filter_remap_rebatch(val_ds)
    test_t = filter_remap_rebatch(test_ds)

    # Quick sanity check on shapes
    for xb, yb in train_t.take(1):
        print("Sanity check shapes - X:", xb.shape, "y:", yb.shape)

    # 4) Performance pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_t = train_t.cache().prefetch(AUTOTUNE)
    val_t = val_t.cache().prefetch(AUTOTUNE)
    test_t = test_t.cache().prefetch(AUTOTUNE)

    # 5) Augmentation
    data_aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation"
    )

    # 6) Model: EfficientNetB0 transfer learning
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
    outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("stageB_tumour_type.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ]

    # 7) Train head
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    print("\n--- Stage B: training classifier head (tumour types only) ---")
    model.fit(val_t.take(0), epochs=0)  # no-op to initialise lookup (harmless)
    model.fit(train_t, validation_data=val_t, epochs=epochs_head, callbacks=callbacks)

    # 8) Fine-tune
    base.trainable = True
    fine_tune_at = int(len(base.layers) * 0.7)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    print("\n--- Stage B: fine-tuning ---")
    model.fit(train_t, validation_data=val_t, epochs=epochs_finetune, callbacks=callbacks)

    # 9) Evaluate tumour-only test set
    print("\n--- Stage B: tumour-only test evaluation ---")
    test_loss, test_acc = model.evaluate(test_t)
    print(f"Tumour-only test accuracy: {test_acc:.4f}")

    # 10) Detailed report
    y_true = np.concatenate([y.numpy() for _, y in test_t], axis=0)
    y_prob = model.predict(test_t)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nStage B classification report (tumour types only):")
    print(classification_report(y_true, y_pred, target_names=tumour_classes, digits=4))

    print("Stage B confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save tumour class names for pipeline use
    with open("stageB_class_names.txt", "w", encoding="utf-8") as f:
        for c in tumour_classes:
            f.write(c + "\n")

    print("\nSaved: stageB_tumour_type.keras and stageB_class_names.txt")


if __name__ == "__main__":
    main()
