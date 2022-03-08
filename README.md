# Brain-Healthy-vs-Tumor-Image-Classification
Uses a convolutional neural network with pretrained base of InceptionResNetV2 with a fully connected neural network to predict brain MRI images for classification of a tumor or healthy brain 

# How to run and test the code:
1. Pull the project files in a local repository
2. Open the project path in pycharm
3. Create a virtual environment and select the python interpreter 
4. Activate the virtual environment in pycharm 
5. To install all essential libraries run command "pip install -r requirements.txt" using the local terminal 
6. Change the paths inside the code according to your directory names 
7. In the app.py file, run the file in pycharm and copy the API url from the output window

# To test an image and get predictions from the model
1. Install and run "Postman" app on the system
2. copy and paste the API url + "/predict" in postman.
3. Select "method" as "POST", select "body" as "raw"
4. Open "https://base64.guru/converter/encode/image" in your web browser, upload the test image and click on "Encode image to base64" (encodes image to be tested into string format) 
5. Copy the image string from output, go to postman and in the body create a dictionary with key as "image" and value as the copied string (in double quotes).
6. Press "send", and it yields the predictions about the test image in the output window of the postman app. 
