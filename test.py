from llama_cpp import Llama

# ✅ GPU 사용 안 함 (n_gpu_layers=0), 메모리 부담 줄이기 (n_ctx=1024)
llm = Llama(model_path="C:/Users/jhh33/Desktop/llmserver2/Trashlyzer/models/llama-3.2-korean-ggachi-1b-instruct-v1-q4_k_m.gguf", n_threads=6,verbose=True)

print("✅ 모델 로드 성공!")
