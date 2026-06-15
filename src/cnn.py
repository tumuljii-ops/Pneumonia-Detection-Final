import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ==========================================================
# CONFIGURATION
# ==========================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

TRAIN_DIR = "dataset/chest_xray/chest_xray/train"
TEST_DIR = "dataset/chest_xray/chest_xray/test"

# ==========================================================
# DATA AUGMENTATION + VALIDATION SPLIT
# ==========================================================

train_datagen = ImageDataGenerator(

    rescale=1.0/255,

    rotation_range=15,

    zoom_range=0.15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    validation_split=0.20
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255
)

# ==========================================================
# TRAIN GENERATOR
# ==========================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training"
)

# ==========================================================
# VALIDATION GENERATOR
# ==========================================================

val_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation"
)

# ==========================================================
# TEST GENERATOR
# ==========================================================

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)

# ==========================================================
# CNN MODEL
# ==========================================================

model = Sequential([

    # ------------------------------------------------------
    # BLOCK 1
    # ------------------------------------------------------

    Conv2D(
        32,
        (3,3),
        padding="same",
        activation="relu",
        input_shape=(224,224,1)
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    # ------------------------------------------------------
    # BLOCK 2
    # ------------------------------------------------------

    Conv2D(
        64,
        (3,3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    # ------------------------------------------------------
    # BLOCK 3
    # ------------------------------------------------------

    Conv2D(
        128,
        (3,3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    # ------------------------------------------------------
    # BLOCK 4
    # ------------------------------------------------------

    Conv2D(
        256,
        (3,3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    # ------------------------------------------------------
    # FEATURE COMPRESSION
    # ------------------------------------------------------

    GlobalAveragePooling2D(),

    # ------------------------------------------------------
    # DENSE HEAD
    # ------------------------------------------------------

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        1,
        activation="sigmoid"
    )
])

# ==========================================================
# MODEL SUMMARY
# ==========================================================

model.summary()

# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

# ==========================================================
# CALLBACKS
# ==========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=1e-6,

    verbose=1
)

checkpoint = ModelCheckpoint(

    "best_custom_cnn.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)

# ==========================================================
# TRAIN MODEL
# ==========================================================

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]
)

# ==========================================================
# EVALUATE
# ==========================================================

loss, accuracy = model.evaluate(
    test_generator
)

print("\nFinal Test Accuracy:")
print(f"{accuracy*100:.2f}%")

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

model.save(
    "final_custom_cnn.keras"
)

print(
    "\nModel Saved Successfully!"
)