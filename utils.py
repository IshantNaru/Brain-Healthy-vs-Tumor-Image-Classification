import base64
import os


def decodeImage(imgstring, fileName):
    os.chdir("C:/Users/Ishant Naru/Desktop/brain_tumor_classifier/test_dir/test")
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())