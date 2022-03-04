import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import imagepreprocess
from model_fitting import modelFitting


train = "brain tumor data/train"
test = "brain tumor data/test"
val = "brain tumor data/val"

# Loading the images and processing them
train_generator, validation_generator = imagepreprocess(train, val)

# training the model
try:
    history = modelFitting(train_generator, validation_generator)
except Exception as e:
    print(e)

# hist_df = pd.DataFrame(history.history)
# #Displaying the results
# try:
#     pd.DataFrame(hist.history).plot(figsize=(10,7))
#     plt.grid(True)
#     plt.gca().set_ylim(0,1)
#     plt.show()
# except Exception as e:
#     print(e)
