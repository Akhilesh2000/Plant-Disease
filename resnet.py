import tensorflow as tf
import keras
import PIL
from keras.applications.resnet import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# Set up data generators for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('dataset/train', target_size=(256, 256), batch_size=32,
                                                    class_mode='categorical')
val_generator = val_datagen.flow_from_directory('dataset/val', target_size=(256, 256), batch_size=32,
                                                class_mode='categorical')

# Create ResNet-50 model with pre-trained weights
resnet_model = Sequential()
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers on top of the base model
resnet_model.add(base_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(15, activation='softmax'))
# x = Flatten()(base_model.output)
# x = Dense(512, activation='relu')(x)
# predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Compile the model with Adam optimizer and categorical cross-entropy loss
# model = Model(inputs=base_model.input, outputs=predictions)
# model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model with fit_generator
# model.fit(train_generator,
#                     steps_per_epoch=len(train_generator),
#                     epochs=10, validation_data=val_generator,
#                     validation_steps=len(val_generator)
#                     )

resnet_model.summary()
resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 5
history = resnet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

import tensorflow as tf
from keras.models import load_model


Model.save('ResNet.h5')

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


