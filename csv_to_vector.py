import csv
import json
from sentence_transformers import SentenceTransformer

# 모델 로드
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

def extract_descriptions(csv_path: str) -> list[str]:
    descriptions = []
    with open(csv_path, "r", encoding="euc-kr") as f:
        reader = csv.reader(f)
        next(reader, None)  # 헤더 스킵
        for row in reader:
            if len(row) < 2:
                continue
            item_text = ", ".join(row[:]).strip()
            descriptions.append(item_text)
    return descriptions

def get_embedding(text: str):
    return model.encode(text).tolist()

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

    with open("item_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print("완료")


run()