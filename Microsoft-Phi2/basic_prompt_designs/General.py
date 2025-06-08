from basic_prompt_designs.Global_Function import Global_Function
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import streamlit as st
import re


class General:
    def __init__(self, tokenizer, model, comp_model, X, vectorizer, task_lib):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model, comp_model, X, vectorizer, task_lib)

    def zero_shot_direct(self, text):
        return text
    def zero_shot_CoT(self, text):
        return text
    def self_consistency(self, prompt, num_samples=5, max_len=150):
        outputs = []
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        input_len = inputs["input_ids"].shape[1]

        for _ in range(num_samples):
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_len,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            new_tokens = output[0][input_len:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(decoded)

        return outputs

    def majority_vote(self, outputs):
        for output in outputs:
            print("Output:\n", output)
            print()

        answers = [self.functions.extract_answer(output) for output in outputs]
        answers = [a for a in answers if a is not None]

        if not answers:
            return None, Counter()

        counts = Counter(answers)
        return counts.most_common(1)[0][0], counts

    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=200, do_print=False):
        prompt = self.zero_shot_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        if do_print:
            st.code(prompt, language='text')
        return samples

    def zero_shot_ToT(self, text):
        return text

    def get_best_thought(self, thoughts, text, do_print=False):
        output = self.select_best_path(thoughts, text, do_print=do_print)
        if len(output) < 3:
            match = re.search(r"\b(\d+)\b", output)
            index = int(match.group(1)) - 1
            if 0 <= index < len(thoughts):
                return thoughts[index]
        else:
            return output

    def expand_thoughts(self, prompt, n=3):
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        input_len = inputs["input_ids"].shape[1]

        outputs = [self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        ) for _ in range(n)]

        return [
            self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
            for output in outputs
        ]

    def recursive_expand_tree(self, prompt, depth, breadth, context, do_print=False, node_counter=[0], max_nodes=30):
        if depth == 0 or node_counter[0] >= max_nodes:
            return [prompt]

        base_prompt = f"""Instruction: Use tree-of-thought mathematics reasoning to explore solutions.

                    Question: {context}

                    Partial Reasoning:
                    {prompt}
                    Now expand with next steps:"""

        expanded = self.expand_thoughts(base_prompt, n=breadth)
        expanded = list(set(expanded))

        if do_print:
            print(f"[Depth {depth}] Base prompt:\n{base_prompt}")
            print(f"[Depth {depth}] Got thoughts:\n", expanded)

        tree = []
        for thought in expanded:
            if node_counter[0] >= max_nodes:
                break
            node_counter[0] += 1
            sub_tree = self.recursive_expand_tree(thought, depth - 1, breadth, context, do_print, node_counter,
                                                  max_nodes)
            tree.extend(sub_tree)
        return tree

    def select_best_path(self, thoughts, text, do_print=False):
        prompt = f"""Instruction: Given a question and several reasoning paths, choose the most logical one.

        Question: {text}

        Candidates:
        {chr(10).join([f"{i + 1}. {t}" for i, t in enumerate(thoughts)])}

        Reply with the best option (one candidate, select thought number) 
        Answer: Explain:"""
        if do_print:
            print('prompt:\n', prompt)
        print(thoughts)
        input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        input_len = input['input_ids'].shape[1]
        output = self.functions.generate_output(type=None, input=input, max_len=200)
        generate_ids = output[0][input_len:]
        answer = self.tokenizer.decode(generate_ids, skip_special_tokens=True).strip()
        return answer

    def zero_shot_ToT_expanded(self, text, depth=2, breadth=2, do_print=False):
        root_prompt = self.zero_shot_ToT(text)
        if do_print:
            print('Root_prompt:\n', root_prompt)
        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, context=text, do_print=do_print,
                                          max_nodes=30)
        best = self.get_best_thought(tree, text, do_print)
        return best

    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.few_shots_ToT(text)
        if do_print:
            print('[Root prompt]')
            print(root_prompt)

        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, context=text, do_print=do_print,
                                          max_nodes=30)
        best_path = self.get_best_thought(tree, text, do_print)
        return best_path

    def few_shots_direct(self, text):
        return text

    def few_shots_CoT(self, text):
        return text
    def few_shots_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        if do_print:
            st.code(prompt, language='text')
        return samples

    def few_shots_ToT(self, text):
        return text

    def run(self, text, do_print=False, type='Direct zero-shot', num_samples=5, max_len=500, depth=2, breadth=3, k=3):
        prompt = None
        if type == 'Zero-shot CoT + Self-consistency':
            max_len = 200
            return self.zero_shot_CoT_SC(text, num_samples, max_len, do_print)
        elif type == 'Few-shots CoT + Self-consistency':
            max_len = 200
            return self.few_shots_CoT_SC(text, num_samples, max_len, do_print)
        elif type == 'Zero-shot ToT expanded':
            return self.zero_shot_ToT_expanded(text, depth, breadth, do_print)
        elif type == 'Few-shots ToT expanded':
            return self.few_shots_ToT_expanded(text, depth, breadth, do_print)
        else:
            if type == 'Direct zero-shot':
                prompt = self.zero_shot_direct(text)
            elif type == 'Zero-shot CoT':
                max_len = 200
                prompt = self.zero_shot_CoT(text)
            elif type == 'Zero-shot ToT':
                prompt = self.zero_shot_ToT(text)
            elif type == 'Direct few-shots':
                prompt = self.few_shots_direct(text)
            elif type == 'Few-shots CoT':
                prompt = self.few_shots_CoT(text)
            elif type == 'Few-shots ToT':
                prompt = self.few_shots_ToT(text)
            elif type == 'Few-shots CoT + ART':
                prompt = text
            if do_print:
                st.markdown("### Prompt:")
                st.code(prompt, language="text")
            input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
            input_len = input['input_ids'].shape[1]
            output = self.functions.generate_output(type=None, input=input, max_len=max_len)
            generate_out = output[0][input_len:]
            return self.tokenizer.decode(generate_out, skip_special_tokens=True)
