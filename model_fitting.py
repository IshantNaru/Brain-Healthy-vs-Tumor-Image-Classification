import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import rmsprop_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from preprocessing import predict_generator


def modelBuilding():
    # Importing Convolutional Base

    conv_base = InceptionResNetV2(weights='imagenet',
                                  include_top=False,
                                  input_shape=(200, 200, 3))
    conv_base.trainable = False

    # Fully Connected Neural Network Head
    cnn_model = keras.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(300, activation='relu'),
        layers.Dense(1, activation='sigmoid')])

    # Compilation with loss function, optimizers and eval metrics
    cnn_model.compile(loss='binary_crossentropy',
                      optimizer=rmsprop_v2.RMSprop(learning_rate=2e-5),
                      metrics=['accuracy'])
    print("Compilation was successful...")
    return cnn_model


def modelFitting(train_gen, val_gen):
    cnn = modelBuilding()
    print(cnn.summary())

    # Creating Checkpoints
    checkpoint_cb = ModelCheckpoint("BrainTumorInception.h5",
                                    save_best_only=True)
    early_stopping_cb = EarlyStopping(min_delta=0.0001,
                                      patience=4,
                                      restore_best_weights=True)
    print("Checkpoints successfully created...")

    try:
        history = cnn.fit(train_gen,
                          steps_per_epoch=56,
                          epochs=30,
                          validation_data=val_gen,
                          validation_steps=46,
                          callbacks=[checkpoint_cb, early_stopping_cb],
                          verbose=1)
        return history
    except Exception as e:
        print(e)


def predictions(imagepath, steps):
    try:
        model1 = load_model("C:/Users/Ishant Naru/Desktop/brain_tumor_classifier/BrainTumorInception.h5")

        pred_generator = predict_generator(imagepath)
        print("Predict generator loading successful")

        result = model1.predict(pred_generator, steps)

        cl = np.round(result)
        class_list = cl.tolist()
        filenames = pred_generator.filenames

        # df = pd.DataFrame({"Name": filenames, "predictions": result.tolist(), "class": class_list})
        return filenames, class_list[0][0]

    except Exception as e:
        print(e)
# class model:
#     def __init__(self, image_file):
#         self.image_file = image_file
#
#     def trained_model(self):
#         image_name = self.image_file
#
#         # loading the trained model
#         model1 = load_model("BrainTumorInception.h5")
#
#         test_image = image.load_img(image_name, color_mode='rgb', target_size=(200, 200))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis=0)
#         result = model1.predict(test_image)
#         print(result)
#
#         if result[0][0] == 1:
#             prediction = 'cancer detected'
#             return [{"image": prediction}]
#         else:
#             prediction = 'no cancer present'
#             return [{"image": prediction}]
