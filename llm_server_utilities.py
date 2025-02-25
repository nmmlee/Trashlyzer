import io
from PIL import Image
import cv2
import base64
import numpy as np

def base64_to_img(original: str):
    # base64 디코딩 후 BytesIO로 감싸서 Image.open에 전달
    img_data = base64.b64decode(original)
    img = Image.open(io.BytesIO(img_data))  # io.BytesIO로 감싸기
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)  # OpenCV 형식으로 변환