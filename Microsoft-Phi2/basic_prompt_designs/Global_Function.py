import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from torch.onnx.symbolic_opset9 import cosine_similarity


class Global_Function:
    def __init__(self, tokenizer, model, comp_model=None, X=None, vectorizer=None, task_lib=None):
        self.tokenizer = tokenizer
        self.model = model
        self.comp_model = comp_model
        self.X = X
        self.vectorizer= vectorizer
        self.task_lib = task_lib
    def extract_labels(self, text): #classification
        matches = re.findall(r'\b(positive|negative|neutral)\b', text.lower())
        return list(set(matches))
    def clean_label(self, text): #classification
        match = re.search(r'\b(positive|negative|neutral)\b', text.lower())
        return match.group(1) if match else 'unknown'

    import re
    import re

    def extract_answer(self, text):
        for line in text.strip().splitlines()[::-1]:
            line_lower = line.lower()
            if re.search(r'\b(remaining|final|total|answer|so|therefore|thus)\b', line_lower):
                matches = re.findall(r'= *\$?(-?\d+(?:\.\d+)?)', line)
                if matches:
                    num = matches[-1]
                    if float(num) > 1:
                        return num
        matches = re.findall(r'= *\$?(-?\d+(?:\.\d+)?)', text)
        if matches:
            num = matches[-1]
            if float(num) > 1:
                return num
        return None

    def extract_target(self, text):
        for line in text.splitlines()[::-1]:
            if line.strip().startswith("####"):
                return line.strip().replace("####", "").strip()
        return None
    def split_cot_and_answer(self, text):
        lines = text.strip().striplines()
        cot_lines = [line for line in lines if not line.strip().startswith("####")]
        answer = self.extract_target(text)
        return "\n".join(cot_lines), answer

    def generate_output(self, type, input, max_len=200):
        if type=='generation':
            output = self.model.generate(
                **input,
                max_new_tokens= max_len,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
            )
        else:
            output = self.model.generate(
                **input,
                max_new_tokens= max_len,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        return output

    def generate_strategy(self, prompt_base, strategy_index, max_len=100, type='generation'):
        prompt = prompt_base + \
            f"Strategy {strategy_index + 1}: \n"
        input = self.tokenizer(prompt, return_tensor='pt').to('cuda')
        output = self.generate_output(type, input, max_len)
        return self.tokenizer.decode(output[0],skip_special_tokens=True)

    def self_consistency(self, prompt, num_samples=5, max_len=50):
        outputs = []
        for i in range(num_samples):
            input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
            input_len = input['input_ids'].shape[1]
            output = self.model.generate(
                **input,
                max_new_tokens = max_len,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generate_ids = output[0][input_len:]
            decoded = self.tokenizer.decode(generate_ids, skip_special_tokens=True)
            outputs.append(decoded)
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
        scores = cosine_similarity(q, self.X).flatten()
        top = scores.argsort()[::-1][:k]
        return [self.task_lib[i] for i in top]

    def expand_thoughts(self, prompt, n=3, max_len=100):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_len = inputs['input_ids'].shape[1]
    
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_len,
            num_return_sequences=n,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
        thoughts = []
        for output in outputs:
            # Lấy phần từ input_len trở đi
            new_tokens = output[input_len:]
            thought = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            thoughts.append(thought)
    
        return thoughts
