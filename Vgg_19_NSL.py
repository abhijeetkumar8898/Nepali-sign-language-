#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D, SpatialDropout2D
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import class_weight


IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = 37
EPOCHS = 50
DATASET_DIR ="/Users/abhijeetkumar/Desktop/GPU/dataset_splitted2"  #Dataset Directory (Contains images of 224,224,3)
MODEL_SAVE_PATH = "/Users/abhijeetkumar/Desktop/GPU/final_vgg19_hand_model4.h5"
WEIGHTS_SAVE_PATH = "/Users/abhijeetkumar/Desktop/GPU/vgg19_hand.weights.h5"
CLASS_INDEX_PATH = "/Users/abhijeetkumar/Desktop/GPU/class_indices.json"

#VGG-19

def build_vgg19(input_shape=(224, 224, 3), num_classes=37):
    inputs = layers.Input(shape=input_shape)
    # Block 1
    x = layers.Conv2D(16, (3, 3), padding='same', activation="relu")(inputs)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = SpatialDropout2D(0.2)(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = SpatialDropout2D(0.2)(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = SpatialDropout2D(0.2)(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 5
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = SpatialDropout2D(0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    # Dense
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATASET_DIR}/train", image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=True)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATASET_DIR}/val", image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=False)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATASET_DIR}/test", image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=False)

    # Save class indices
    class_indices_dict = {i: name for i, name in enumerate(train_ds.class_names)}
    with open(CLASS_INDEX_PATH, 'w') as f:
        json.dump(class_indices_dict, f)

    norm = layers.Rescaling(1./255)
    aug = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1)
    ])

    train_ds = train_ds.map(lambda x, y: (norm(aug(x, training=True)), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

def compute_class_weights(train_ds):
    labels = [label.numpy() for _, label in train_ds.unbatch()]
    weights = class_weight.compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=labels)
    return dict(enumerate(weights))

def train_model(model, train_ds, val_ds, class_weights):
    if not os.path.exists(os.path.dirname(WEIGHTS_SAVE_PATH)):
        os.makedirs(os.path.dirname(WEIGHTS_SAVE_PATH))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(filepath=WEIGHTS_SAVE_PATH, save_weights_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                        class_weight=class_weights, callbacks=callbacks)
    model.save(MODEL_SAVE_PATH)
    return history

def evaluate_model(model, test_ds):
    loss, acc = model.evaluate(test_ds)
    print(f" Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.show()

def evaluate_metrics(model, test_ds):
    with open(CLASS_INDEX_PATH, 'r') as f:
        class_indices = json.load(f)
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = np.argmax(model.predict(images), axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print(f"\n Macro-Averaged Scores:")
    print(f"F1 Score    : {f1:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n Confusion Matrix:\n{cm}")

    iou_per_class = []
    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        denom = TP + FP + FN
        iou = TP / denom if denom != 0 else 0
        iou_per_class.append(iou)

    mean_iou = np.mean(iou_per_class)
    print(f"\n Mean IoU (macro-averaged): {mean_iou:.4f}")
    for i, iou in enumerate(iou_per_class):
        print(f"Class {i} ({class_names[i]}): {iou:.4f}")


if __name__ == "__main__":
    print(" Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets()

    print(" Computing class weights...")
    weights = compute_class_weights(train_ds)

    print(" Building model...")
    model = build_vgg19()

    print(" Training model...")
    history = train_model(model, train_ds, val_ds, weights)

    print(" Evaluating model on test set...")
    evaluate_model(model, test_ds)

    print(" Plotting training history...")
    plot_history(history)

    print(" Evaluating classification metrics...")
    evaluate_metrics(model, test_ds)



