import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import requests
from io import BytesIO
import time

def extract_features(image):
    features=[]
    response = requests.get(image)
    test_path=BytesIO(response.content)
    img=Image.open(test_path)
    img.save('test.jpg')
    im=cv2.imread('test.jpg',1)
    im=cv2.resize(im,(180,180),interpolation=cv2.INTER_AREA)
    im=np.array(im)
    features.append(im)
    features=np.array(features)
    features=features.reshape(1,180,180,3)
    features=features/255.0
    features=np.float16(features)
    return features

def check_thumbnail(model, thumbnail_id, output):

    features=extract_features(thumbnail_id)

    pr_start=time.time()
    op=model.predict(features)
    output['thumbnail_a']=op[0][0]
    # return op[0][0]

