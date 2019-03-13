import tensorflow as tf
from keras.models import  Model
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,Lambda,Flatten,Activation, Input , BatchNormalization
from keras.optimizers import Adam

from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.visible_device_list = "0"
#set_session(tf.Session(config=config))

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs

input_data = Input(name='the_input', shape=(32,32,3), dtype='float32')
    
inner = Conv2D(96, (9,9), padding='valid',name='conv1')(input_data)

inner = BatchNormalization()(inner)

inner=Lambda(max_out,arguments={'num_units':48},name='maxout1')(inner)

inner = Conv2D(128, (9,9), padding='valid',name='conv2')(inner)

inner = BatchNormalization()(inner)

inner=Lambda(max_out,arguments={'num_units':64},name='maxout2')(inner)

inner = Conv2D(256, (9,9), padding='valid',name='conv3')(inner)

inner = BatchNormalization()(inner)

inner=Lambda(max_out,arguments={'num_units':128},name='maxout3')(inner)

inner = Conv2D(512, (8,8), padding='valid',name='conv4')(inner)

inner = BatchNormalization()(inner)

output=Lambda(max_out,arguments={'num_units':128},name='maxout4')(inner)


inner = Conv2D(12540, (1,1), padding='valid',name='conv5_2')(inner)

inner=Lambda(max_out,arguments={'num_units':3135},name='maxout5_2')(inner)

flatten=Flatten()(inner)

output2=Activation('softmax')(flatten)

model = Model(input_data, output2)

#model.summary()

model.load_weights("model3.hdf5",by_name=True)
for layer in model.layers:
	layer.trainable = False
for layer in model.layers[-4:]:
        layer.trainable = True 
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

batch_size = 512
train_datagen = ImageDataGenerator(zoom_range=0.1)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '../data/ETL_data_all',  # this is the target directory
        target_size=(32, 32),  # all images will be resized to 150x150
        batch_size=batch_size)
validation_generator = test_datagen.flow_from_directory(
        '../data/ETL_data_all_test',  # this is the target directory
        target_size=(32, 32),  # all images will be resized to 150x150
        batch_size=batch_size)
label_map = (train_generator.class_indices)
file =open("testfile2.txt","w")
file.write(json.dumps(label_map))
file.close()
model.fit_generator(
        train_generator,
        steps_per_epoch=1000000 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=400000 // batch_size)

model.save("first_try_maxout.h5")

#model.save_weights("model4.hdf5")
