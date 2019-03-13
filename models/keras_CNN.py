import tensorflow as tf
from keras.models import Model
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Lambda, Flatten, Activation, Input, BatchNormalization, MaxPooling2D, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

input_data = Input(name='the_input', shape=(32, 32, 3), dtype='float32')
inner = Conv2D(16, (3, 3), padding='same', activation='relu', data_format="channels_last", name='conv1')(input_data)
inner = BatchNormalization()(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='pool1')(inner)
inner = Conv2D(32, (3, 3), padding='same', activation='relu', data_format="channels_last", name='conv2')(inner)
inner = BatchNormalization()(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='pool2')(inner)
inner = Conv2D(64, (3, 3), padding='same', activation='relu', data_format="channels_last", name='conv3')(inner)
inner = BatchNormalization()(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='pool3')(inner)
inner = Conv2D(64, (3, 3), padding='same', activation='relu', data_format="channels_last", name='conv4')(inner)
inner = BatchNormalization()(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='pool4')(inner)
inner = Conv2D(128, (2, 2), padding='same', activation='relu', data_format="channels_last", name='conv5')(inner)
print inner.shape
inner = BatchNormalization()(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='pool5')(inner)
flatten = Flatten()(inner)
output = Dense(3135, activation='softmax')(flatten)
model = Model(input_data, output)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
batch_size = 64
train_datagen = ImageDataGenerator(rotation_range=5, shear_range=0.2, zoom_range=0.2)
test_datagen = ImageDataGenerator(rotation_range=5, shear_range=0.2, zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(
    '../data/ETL_data_all',
    target_size=(32, 32),
    batch_size=batch_size)
validation_generator = test_datagen.flow_from_directory(
    '../data/ETL_data_all_test',
    target_size=(32, 32),
    batch_size=batch_size)
label_map = (train_generator.class_indices)
file = open("testfileCNN.txt", "w")
file.write(json.dumps(label_map))
file.close()
save_dir = "../models/checkpoint_CNN"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_path = save_dir + "/"  + 'weights-{epoch:02d}-{val_loss:.2f}.hdf5'
check_pointer = ModelCheckpoint(save_path, save_best_only=True)
model.fit_generator(
    train_generator,
    steps_per_epoch=1000000 // batch_size,
    epochs=5,
    callbacks=[check_pointer],
    validation_data=validation_generator,
    validation_steps=400000 // batch_size)
model.save_weights("first_try_cnn_1.hdf5")
