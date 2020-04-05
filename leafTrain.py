from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense, Activation, ZeroPadding2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping


img_rows = 224
img_cols = 224
num_classes = 40

vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(img_rows, img_cols,3))


for layer in vgg16.layers:
    layer.trainable = False

# for (i,layer) in enumerate(vgg16.layers):
#     print(str(i)+" "+ layer.__class__.__name__, layer.trainable)

def addtop_model(bottom_model, num_classes, D=256):
    
    top_model = bottom_model.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(D, activation="relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes,activation="softmax")(top_model)
    return top_model

FC_Head = addtop_model(vgg16, num_classes)

model = Model(inputs=vgg16.input, outputs= FC_Head)

print(model.summary())

train_dir = "./dataset/train"
test_dir = "./dataset/test"

train_datagen = ImageDataGenerator(
    rescale=1. /255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip= True,
    fill_mode="nearest",
    rotation_range=20
)

test_datagen  = ImageDataGenerator(rescale=1. /255)

train_batchsize = 16
test_batchsize = 8

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size= train_batchsize,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    batch_size= test_batchsize,
    class_mode="categorical",
    shuffle=False
)

checkpoint = ModelCheckpoint(
    "leafs.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

early = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=1,
    restore_best_weights= True
)

callbacks = [checkpoint, early]

model.compile(loss="categorical_crossentropy", optimizer= RMSprop(lr=0.001), metrics=["accuracy"])

nb_train = 323
nb_test = 121
batch_size=4
epochs = 30

model.fit(
    train_generator,
    steps_per_epoch= nb_train // batch_size,
    epochs= epochs,
    callbacks=callbacks,
    validation_data= test_generator,
    validation_steps= nb_test // batch_size
)