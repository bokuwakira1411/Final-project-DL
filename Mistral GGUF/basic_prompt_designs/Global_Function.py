import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util

class Global_Function:
    def __init__(self, model, comp_model=None, X=None, vectorizer=None, task_lib=None):
        self.model = model  
        self.comp_model = comp_model
        self.X = X
        self.vectorizer = vectorizer
        self.task_lib = task_lib

    def extract_labels(self, text):
        matches = re.findall(r'\b(positive|negative|neutral)\b', text.lower())
        return list(set(matches))

    def clean_label(self, text):
        match = re.search(r'\b(positive|negative|neutral)\b', text.lower())
        return match.group(1) if match else 'unknown'

    def extract_answer(self, text):
        matches = re.findall(r'= *\$?(-?\d+(?:\.\d+)?)', text)
        return matches[-1] if matches else None

    def extract_target(self, text):
        for line in reversed(text.splitlines()):
            if line.strip().startswith("####"):
                return line.strip().replace("####", "").strip()
        return None

    def split_cot_and_answer(self, text):
        lines = text.strip().splitlines()
        cot_lines = [line for line in lines if not line.strip().startswith("####")]
        answer = self.extract_target(text)
        return "\n".join(cot_lines), answer

    def generate_output(self, prompt):
        response = self.model.chat(messages=[
            {"role": "user", "content": prompt}
        ])
        return response["choices"][0]["message"]["content"]

    def generate_strategy(self, prompt_base, strategy_index):
        prompt = prompt_base + f"Strategy {strategy_index + 1}:\n"
        return self.generate_output(prompt)

    def self_consistency(self, prompt, num_samples=5):
        outputs = []
        for _ in range(num_samples):
            outputs.append(self.generate_output(prompt))
        return outputs

    def majority_vote(self, outputs):
        counts = Counter(outputs)
        return counts.most_common(1)[0][0], counts

    def cosine_similarity_score(self, pred, target):
        emb1 = self.comp_model.encode(pred, convert_to_tensor=True)
        emb2 = self.comp_model.encode(target, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2)[0][0])

    def find_top_k_tasks(self, text, k=3):
        q = self.vectorizer.transform([text])
        scores = (q @ self.X.T).flatten()
        top = scores.argsort()[::-1][:k]
        return [self.task_lib[i] for i in top]

    def expand_thoughts(self, prompt, n=3):
        thoughts = []
        for _ in range(n):
            thoughts.append(self.generate_output(prompt))
        return thoughts
