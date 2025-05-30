# Mistral/GGUFClient.py
from llama_cpp import Llama

class GGUFClient:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)

    def chat(self, messages):
        prompt = ""
        for m in messages:
            if m["role"] == "user":
                prompt += f"[INST] {m['content']} [/INST]"
            elif m["role"] == "assistant":
                prompt += m["content"]
        output = self.llm(prompt, max_tokens=512)
        return {
            "choices": [
                {
                    "message": {
                        "content": output["choices"][0]["text"]
                    }
                }
            ]
        }
