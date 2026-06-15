import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

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

EPOCHS = 10

TRAIN_DIR = "dataset/chest_xray/chest_xray/train"

TEST_DIR = "dataset/chest_xray/chest_xray/test"

# ==========================================================
# DATA GENERATORS
# ==========================================================

train_datagen = ImageDataGenerator(

    rescale=1.0/255,

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255
)

# ==========================================================
# TRAIN DATA
# ==========================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="rgb",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training"
)

# ==========================================================
# VALIDATION DATA
# ==========================================================

val_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="rgb",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation"
)

# ==========================================================
# TEST DATA
# ==========================================================

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="rgb",

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)

# ==========================================================
# LOAD MOBILENETV2
# ==========================================================

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)
)

# ==========================================================
# FREEZE PRETRAINED LAYERS
# ==========================================================

base_model.trainable = False

# ==========================================================
# CUSTOM CLASSIFIER HEAD
# ==========================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

output = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(

    inputs=base_model.input,

    outputs=output
)

# ==========================================================
# MODEL SUMMARY
# ==========================================================

model.summary()

# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

# ==========================================================
# CALLBACKS
# ==========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=3,

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

    "best_mobilenetv2.keras",

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
# TEST ACCURACY
# ==========================================================

loss, accuracy = model.evaluate(
    test_generator
)

print("\nFinal Test Accuracy:")
print(f"{accuracy*100:.2f}%")

# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    "final_mobilenetv2.keras"
)

print(
    "\nModel Saved Successfully!"
)