import io
from PIL import Image
import cv2
import base64
import numpy as np
import json
import time
import csv

def base64_to_img(original: str):
    # base64 디코딩 후 BytesIO로 감싸서 Image.open에 전달
    img_data = base64.b64decode(original)
    img = Image.open(io.BytesIO(img_data))  # io.BytesIO로 감싸기
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)  # OpenCV 형식으로 변환
'''
def extract_descriptions(csv_path: str) -> list[str]:
    descriptions = []
    with open(csv_path, "r", encoding="euc-kr") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            # 가격 제외하고 의미만
            item_text = ", ".join(row[:-1]).strip()
            descriptions.append(item_text)
    return descriptions


def get_embedding(text: str):
    response = client.embeddings.create(
        input=[text],  # 리스트 형태로 넣어야됨
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def run():
    descriptions = extract_descriptions("./data/대형폐기물분류표_노원_crawler.csv")
    results = []

    for i, desc in enumerate(descriptions):
        print(f"[{i+1}/{len(descriptions)}] embedding: {desc}")
        try:
            vector = get_embedding(desc)
            results.append({
                "text": desc,
                "embedding": vector
            })
        except Exception as e:
            print(f"error: {e}")
        time.sleep(1.01)  # 1분에 최대 요청 60개임

    with open("item_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print("완료")
'''