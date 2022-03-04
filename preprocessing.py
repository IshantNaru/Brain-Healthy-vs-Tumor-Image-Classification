import splitfolders as sf
from keras.preprocessing.image import ImageDataGenerator as IDG


def splittingfolders(path, output):
    try:
        sf.ratio(input=path, output=output, seed=1337, ratio=(0.6, 0.2, 0.2))
    except Exception as e:
        print(e)
    else:
        print("Successfully split the folders")


def imagepreprocess(train, val):
    # Creating Image Data Generator train and test objects
    train_datagen = IDG(rescale=1./255,
                        rotation_range=0.1,
                        width_shift_range=0.3,
                        height_shift_range=0.3,
                        zoom_range=0.4,
                        )
    test_datagen = IDG(rescale=1./255)

    # Creating train and validation generators
    try:
        train_generator = train_datagen.flow_from_directory(train,
                                                            target_size=(200, 200),
                                                            batch_size=50,
                                                            class_mode="binary",
                                                            color_mode='rgb')

        validation_generator = test_datagen.flow_from_directory(val,
                                                                target_size=(200, 200),
                                                                batch_size=20,
                                                                class_mode="binary",
                                                                color_mode='rgb')
        return train_generator, validation_generator

    except Exception as e:
        print("Some error loading the images")


def predict_generator(image):
    try:
        pred_generator = IDG(1./255).flow_from_directory(image,
                                                         batch_size=1,
                                                         target_size=(200, 200),
                                                         color_mode='rgb',
                                                         class_mode=None,
                                                         interpolation='nearest',
                                                         shuffle=False,
                                                         seed=45
                                                         )
        return pred_generator
    except Exception as e:
        print(e)



