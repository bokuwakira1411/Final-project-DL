from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import streamlit as st
import re
class Computation(Pattern):
    def __init__(self, tokenizer, model, comp_model, X, vectorizer, task_lib):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model, comp_model, X, vectorizer, task_lib)

    @overrides()
    def zero_shot_direct(self, text):
        return f"""Instruction: Solve the following problem. Problem: {text}              
                Answer:"""
    @overrides()
    def zero_shot_CoT(self, text):
        return f"""Instruction:Solve the following problem step-by-step using formulas, and clearly show your reasoning. Problem: {text}, end when you offer the final answer, do not adding some irrelevant information
                    
                    
                    Final answer: Let's think step by step."""

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
                pad_token_id= self.tokenizer.eos_token_id
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

    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=200, do_print=False):
        prompt = self.zero_shot_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        if do_print:
          st.code(prompt, language='text')
        return samples

    @overrides()
    def zero_shot_ToT(self, text):
        return f"""Instruction: You are a math assistant. For the question below, brainstorm multiple reasoning paths before concluding the answer.
                   Problem: {text}
                   Answer: """


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
                    sub_tree = self.recursive_expand_tree(thought, depth - 1, breadth, context, do_print, node_counter, max_nodes)
                    tree.extend(sub_tree)
                return tree

    def select_best_path(self, thoughts, text, do_print=False):
        prompt = f"""Instruction: You are a math assistant. Given a question and several mathematics reasoning paths, choose the most logical one.

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
        output = self.functions.generate_output(type=None, input = input, max_len = 200)
        generate_ids = output[0][input_len:]
        answer = self.tokenizer.decode(generate_ids, skip_special_tokens=True).strip()
        return answer

    def zero_shot_ToT_expanded(self, text, depth=2, breadth=2, do_print=False):
        root_prompt = self.zero_shot_ToT(text)
        if do_print:
                print('Root_prompt:\n', root_prompt)
        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, context=text, do_print=do_print, max_nodes=30)
        best = self.get_best_thought(tree, text, do_print)
        return best

    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.few_shots_ToT(text)
        if do_print:
            print('[Root prompt]')
            print(root_prompt)

        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, context=text, do_print=do_print, max_nodes=30)
        best_path = self.get_best_thought(tree, text, do_print)
        return best_path

    @overrides()
    def few_shots_direct(self, text):
        return f"""
        Instruction: Solve the problem clearly. Here are some examples: 
        Problem: A store offers a 30% discount on a $200 item. What's the price after discount?
        Step 1: Calculate 30% of 200 = 0.3 * 200 = 60
        Step 2: Subtract from 200 → 200 - 60 = 140
        Answer: $140
        
        Problem: If a car travels 60 miles in 1.5 hours, what is its average speed in miles per hour?
        Step 1: Use the formula: speed = distance / time
        Step 2: speed = 60 / 1.5 = 40
        Answer: 40 miles per hour
        
        Problem: John has 3 boxes. Each box contains 12 apples. He gives away 10 apples. How many apples does he have left?
        Step 1: Total apples = 3 * 12 = 36
        Step 2: Apples left = 36 - 10 = 26
        Answer: 26 apples
        Now solve this problem:
        Problem: {text} 
        Final Answer: 
        """

    @overrides()
    def few_shots_CoT(self, text):
        return f"""
        Instruction: Solve the problem step by step clearly. Here are examples:
        
        Problem: A store offers a 30% discount on a $200 item. What's the price after discount?
        Step 1: Calculate 30% of 200 = 0.3 * 200 = 60
        Step 2: Subtract from 200 → 200 - 60 = 140
        Answer: $140
        
        Problem: If a car travels 60 miles in 1.5 hours, what is its average speed in miles per hour?
        Step 1: Use the formula: speed = distance / time
        Step 2: speed = 60 / 1.5 = 40
        Answer: 40 miles per hour
        
        Problem: John has 3 boxes. Each box contains 12 apples. He gives away 10 apples. How many apples does he have left?
        Step 1: Total apples = 3 * 12 = 36
        Step 2: Apples left = 36 - 10 = 26
        Answer: 26 apples

        Now solve this problem:
        Problem: {text}
        Let's think step by step
        Answer:  """

    @overrides()
    def few_shots_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        if do_print:
            st.code(prompt, language='text')
        return samples

    @overrides()
    def few_shots_ToT(self, text):
        return f"""
        Instruction: Use a tree-of-thought approach to break down complex problems by exploring different solution paths and reasoning step by step, then converge on the correct answer. See the example below:

        Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?  
        Answer:  
        Thought 1: Natalia sold 48 clips in April.  
        Thought 2: She sold half as many in May → CALC(48 / 2) = 24 clips.  
        Thought 3: Total clips sold = CALC(48 + 24) = 72.  
        The final answer is 72.

        Now try the following:  
        Problem: {text}  
        Let's think step by step, exploring each thought:  
        Thought 1:  
        Thought 2:  
        Thought 3: 
        Answer:  """

    def build_prompt(self, examples, query):
        prompt = ""
        for ex in examples:
            prompt += f"""
            Instruction: Solve the mathematic problem
            Problem: {ex['input']}
            Answer: {ex['output']}
            """
            prompt += (
                "Now solve this task:\n"
                f"Problem: {query}\n"
                "Answer:\n"
                "Let's solve this step-by-step. Write your reasoning clearly."
            )
        return prompt

    def few_shots_CoT_ART(self, text, k=3):
        examples = self.functions.find_top_k_tasks(text, k)
        return self.build_prompt(examples, text)


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
                max_len =200
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
                prompt = self.few_shots_CoT_ART(text, k)
            if do_print:
                st.markdown("### Prompt:")
                st.code(prompt, language="text")
            input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
            input_len = input['input_ids'].shape[1]
            output = self.functions.generate_output(type=None, input=input, max_len=max_len)
            generate_out = output[0][input_len:]
            return self.tokenizer.decode(generate_out, skip_special_tokens=True)
