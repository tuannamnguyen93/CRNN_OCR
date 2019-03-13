import tensorflow as tf
from keras.models import Model
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Flatten, Activation, Input, Dropout, Dense
from keras.optimizers import Adam

# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config))

model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
model.summary()
last_layer = model.get_layer('block5_pool').output

x = Flatten(name='flatten')(last_layer)
x = Dropout(0.1)(x)
x = Dense(512, activation='relu', name='fc8')(x)
x = Dropout(0.1)(x)
out = Dense(3137, activation='softmax', name='prediction')(x)
model_1 = Model(model.input, out)
for layer in model_1.layers:
    layer.trainable = False
for layer in model_1.layers[-4:]:
    layer.trainable = True

model_1.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
batch_size = 64
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=5, shear_range=0.2, zoom_range=0.2)
test_datagen = ImageDataGenerator(rotation_range=5, shear_range=0.2, zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(
    '../data/ETL_data_all',  # this is the target directory
    target_size=(32, 32),  # all images will be resized to 150x150
    batch_size=batch_size)
validation_generator = test_datagen.flow_from_directory(
    '../data/ETL_data_test',  # this is the target directory
    target_size=(32, 32),  # all images will be resized to 150x150
    batch_size=batch_size)
label_map = (train_generator.class_indices)
file = open("testfile2.txt", "w")
file.write(json.dumps(label_map))
file.close()
model_1.fit_generator(
    train_generator,
    steps_per_epoch=200000 // batch_size,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=20000 // batch_size)
model_1.save("first_try_VGG16.h5")
