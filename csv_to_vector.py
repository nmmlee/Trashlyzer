import csv
import json
import google.generativeai as genai

# Gemini API 키 설정
try:
    with open("api_token.json", "r") as f:
        config = json.load(f)
        token = config["gemini_api"]["token"]
        genai.configure(api_key=token)
except FileNotFoundError:
    print("api_token.json 파일을 찾을 수 없습니다. API 키를 설정해주세요.")
    exit()


def extract_descriptions(csv_path: str) -> list[str]:
    descriptions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 헤더 스킵
        for row in reader:
            if len(row) < 2:
                continue
            item_text = ", ".join(row[:]).strip()
            descriptions.append(item_text)
    return descriptions

def get_embedding(text: str, model_name="models/embedding-001"):
    return genai.embed_content(
        model=model_name,
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )["embedding"]

def run():
    descriptions = extract_descriptions("./data/대형폐기물분류표_노원_crawler.csv")
    results = []

    for i, desc in enumerate(descriptions):
        desc = desc.replace("(1599-0903 단일 무상 수거 품목)", "")
        print(f"[{i+1}/{len(descriptions)}] embedding: {desc}")
        try:
            vector = get_embedding(desc)
            results.append({
                "text": desc,
                "embedding": vector
            })
        except Exception as e:
            print(f"error: {e}")

    with open("./data/대형폐기물분류표_vectorized_gemini.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print("완료")

if __name__ == "__main__":
    run()
