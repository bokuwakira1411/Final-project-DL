from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
import streamlit as st

class Computation(Pattern):
    def __init__(self, model, comp_model=None, X=None, vectorizer=None, task_lib=None):
        self.functions = Global_Function(model, comp_model, X, vectorizer, task_lib)

    @overrides()
    def zero_shot_direct(self, text):
        return f"""You are a helpful and smart math assistant.
            
            Solve the following problem step-by-step using formulas, and clearly show your reasoning.
            
            Problem: {text}
            
            Let's think step by step.
            
            Answer:"""

    @overrides()
    def zero_shot_CoT(self, text):
        return return f"""You are a helpful and smart math assistant.

                    Solve the following problem step-by-step using formulas, and clearly show your reasoning.
                    
                    Problem: {text}
                    
                    Let's think step by step.
                    
                    Answer:"""

    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples)
        best_answer, all_votes = self.functions.majority_vote(samples)

        if do_print:
            st.markdown("### 📝 Answers (Self-consistency)")
            st.code("\n".join(samples), language="text")
            st.markdown("**Best Answer:**")
            st.code(best_answer)
            st.markdown("**All Votes:**")
            st.code(str(all_votes))

        return best_answer, all_votes

    @overrides()
    def zero_shot_ToT(self, text):
        return f"""Explore multiple ways to solve this math problem:
Question: {text}
Thought 1:
Thought 2:
Thought 3:
Final Answer:"""

    def select_best_path(self, thoughts, text, do_print):
        prompt = f"""
You are a math expert. Given a question and several solution paths, choose the best one.
Question: {text}
Options:
{chr(10).join([f"{i + 1}. {t}" for i, t in enumerate(thoughts)])}
Give the number and explain why it is best. Always calculate all formulas.
"""
        if do_print:
            st.markdown("### Best Path Selection Prompt:")
            st.code(prompt, language="text")

        return self.functions.generate_output(prompt)

    def recursive_expand_tree(self, prompt, depth, breadth, do_print=False):
        if depth == 0:
            return [prompt]
        expanded = self.functions.expand_thoughts(prompt, n=breadth)
        if do_print:
            st.markdown(f"### Depth {depth} Expanded Prompt")
            st.code(prompt)
            st.markdown("**Expanded Thoughts:**")
            st.code("\n\n".join(expanded))

        tree = []
        for thought in expanded:
            tree.extend(self.recursive_expand_tree(thought, depth - 1, breadth, do_print))
        return tree

    def zero_shot_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.zero_shot_ToT(text)
        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
        return self.select_best_path(tree, text, do_print)

    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.few_shots_ToT(text)
        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
        return self.select_best_path(tree, text, do_print)

    @overrides()
    def few_shots_direct(self, text):
        return f"""
You are a helpful and accurate math assistant. Whenever you see a math expression, evaluate it using the function CALC(). Always write all intermediate steps explicitly.

Here are some examples:

Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold CALC(48/2) = 24 clips in May.
Total clips = CALC(48 + 24) = 72.
#### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: Weng earns CALC(12 / 60) = 0.2 dollars per minute.
In 50 minutes, she earned CALC(0.2 * 50) = 10 dollars.
#### 10

Q: {text}
A: Let's solve this step by step. Remember to use CALC() to evaluate all math.
"""

    @overrides()
    def few_shots_CoT(self, text):
        return f"""Solve these math problems step by step:

Q1: A store offers a 30% discount on a $200 item. What's the price after discount?
Step 1: Calculate 30% of 200 = 0.3 * 200 = 60
Step 2: Subtract from 200 → 200 - 60 = 140
A1: $140

Q2: If a car travels 60 miles in 1.5 hours, what is its average speed in miles per hour?
Step 1: Use the formula: speed = distance / time
Step 2: speed = 60 / 1.5 = 40
A2: 40 miles per hour

Q3: John has 3 boxes. Each box contains 12 apples. He gives away 10 apples. How many apples does he have left?
Step 1: Total apples = 3 * 12 = 36
Step 2: Apples left = 36 - 10 = 26
A3: 26 apples

Q4: A shirt originally costs $80. It is marked up by 25%. What's the new price?
Step 1: Calculate markup = 25% of 80 = 0.25 * 80 = 20
Step 2: New price = 80 + 20 = 100
A4: $100

Now solve:
{text}
 Let's solve this step by step. Remember to evaluate all math.
"""

    @overrides()
    def few_shots_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples)
        best_answer, all_votes = self.functions.majority_vote(samples)

        if do_print:
            st.markdown("### 📝 Answers (Self-consistency)")
            st.code("\n".join(samples), language="text")
            st.markdown("**Best Answer:**")
            st.code(best_answer)
            st.markdown("**All Votes:**")
            st.code(str(all_votes))

        return best_answer, all_votes

    @overrides()
    def few_shots_ToT(self, text):
        return f"""Explore multiple ways to solve this math problem:
Q: A man spends 1/3 of his salary on rent, 1/4 on food, and saves the rest. What fraction does he save?
Thought 1: 1/3 + 1/4 = (4+3)/12 = 7/12 spent → saves 5/12  
Thought 2: Total spent = 7/12 → leftover = 5/12  
Thought 3: Validate: 1 - 7/12 = 5/12  
Final Answer: 5/12

Now try this:
{text}
Thought 1:
Thought 2:
Thought 3:
Final Answer:"""

    def build_prompt(self, examples, query):
        prompt = ""
        for ex in examples:
            prompt += f"Task: Math\nInput: {ex['input']}\nOutput: {ex['output']}\n"
        prompt += f"\nNow solve this:\nInput: {query}\nOutput:"
        return prompt

    def few_shots_CoT_ART(self, text, k=3):
        examples = self.functions.find_top_k_tasks(text, k)
        return self.build_prompt(examples, text)

    def run(self, text, do_print=False, type='Direct zero-shot', num_samples=5, max_len=50, depth=2, breadth=3, k=3):
        if type == 'Zero-shot CoT + Self-consistency':
            return self.zero_shot_CoT_SC(text, num_samples, max_len, do_print)
        elif type == 'Few-shots CoT + Self-consistency':
            return self.few_shots_CoT_SC(text, num_samples, max_len, do_print)
        elif type == 'Zero-shot ToT expanded':
            return self.zero_shot_ToT_expanded(text, depth, breadth, do_print)
        elif type == 'Few-shots ToT expanded':
            return self.few_shots_ToT_expanded(text, depth, breadth, do_print)
        else:
            if type == 'Direct zero-shot':
                prompt = self.zero_shot_direct(text)
            elif type == 'Zero-shot CoT':
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

            return self.functions.generate_output(prompt)
