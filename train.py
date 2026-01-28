import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import kagglehub
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # Reproducibility
    tf.keras.utils.set_random_seed(42)

    # 1) Download dataset via KaggleHub (your exact line)
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

    # 3) Load datasets (Training is split into train/val; Testing is held out)
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
    num_classes = len(class_names)
    print("Classes:", class_names)


    # ---- Solution 2: Class weights (ADD HERE) ----
    # IMPORTANT: compute weights BEFORE caching/prefetching,
    # so we can safely iterate labels.
    y_train = np.concatenate([y.numpy() for _, y in train_ds], axis=0)

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train
    )
    alpha = 0.3  # 0 = no weights, 1 = full balanced weights
    class_weight = {i: float(1.0 + alpha * (w - 1.0)) for i, w in enumerate(class_weights_arr)}
    print("Soft class weights:", class_weight)

    # 4) Performance pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 5) Data augmentation
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
    base.trainable = False  # freeze base for first stage

    inputs = keras.Input(shape=img_size + (3,))
    x = data_aug(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    # 7) Compile and train (head)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("brain_tumour_model.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ]

    print("\n--- Training classifier head ---")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_head, callbacks=callbacks, class_weight=class_weight)

    # 8) Fine-tune: unfreeze top part of base
    base.trainable = True
    # Freeze early layers, unfreeze later layers
    fine_tune_at = int(len(base.layers) * 0.7)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    print("\n--- Fine-tuning ---")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune, callbacks=callbacks, class_weight=class_weight)

    # 9) Evaluate on test set
    print("\n--- Test evaluation ---")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")

    # 10) Detailed metrics (confusion matrix, report)
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_prob = model.predict(test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save class names for inference
    with open("class_names.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")

    print("\nSaved: brain_tumour_model.keras and class_names.txt")

if __name__ == "__main__":
    main()
