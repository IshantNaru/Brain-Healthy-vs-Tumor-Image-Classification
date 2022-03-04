from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS, cross_origin
from utils import decodeImage
from model_fitting import predictions


app = Flask(__name__)


# class ClientApp:
#     def __init__(self):
#         self.filename = "Cancer_or_No_cancer.jpg"
#         self.classifier = model(image_file=self.filename).trained_model()


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predictRoute():
    image = request.json["file"]
    filename = 'new image.jpg'
    decodeImage(image, filename)
    imagepath = "C:/Users/Ishant Naru/Desktop/brain_tumor_classifier/test_dir"
    file, pred_result = predictions(imagepath, steps=1)
    if pred_result == 0.0:
        status = 'Cancer Detected'
    else:
        status = 'No cancer'
    return f'The image {file} uploaded has a class: {pred_result}, which means - {status}'


if __name__ == "__main__":
    app.run(debug=True)
