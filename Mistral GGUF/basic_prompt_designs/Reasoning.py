from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
import streamlit as st

class Reasoning(Pattern):
    def __init__(self, model, comp_model=None, X=None, vectorizer=None, task_lib=None):
        self.functions = Global_Function(model, comp_model, X, vectorizer, task_lib)

    @overrides()
    def zero_shot_direct(self, text):
        return f"You are a reasoning assistant. Provide a direct answer to the following question:\n{text}"

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""You are a reasoning assistant. Think step by step to answer this question:
{text}
Reasoning:"""

    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.zero_shot_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples)
        best_answer, all_votes = self.functions.majority_vote(samples)

        if do_print:
            st.markdown("### 🧠 Self-consistency Samples")
            st.code("\n".join(samples), language="text")
            st.markdown("**Best Answer:**")
            st.code(best_answer)
            st.markdown("**All Votes:**")
            st.code(str(all_votes))

        return best_answer, all_votes

    @overrides()
    def zero_shot_ToT(self, text):
        return f"""You are a reasoning expert. Explore three possible interpretations to solve this:
Question: {text}
Thought 1:
Thought 2:
Thought 3:
Final Answer:"""

    @overrides()
    def few_shots_direct(self, text):
        return f"""Answer the following reasoning questions directly:
Q: A student studied hard but failed the test. Why might that be?
A: He may have studied the wrong material or was anxious.

Q: {text}
A:"""

    @overrides()
    def few_shots_CoT(self, text):
        return f"""Think step-by-step to answer this question:
Q: A man sees a broken window and a baseball inside the room. What happened?
Step 1: Baseball and broken window suggest it broke the glass  
Step 2: Someone hit a baseball through the window  
A: The window was broken by a baseball

Now solve:
Q: {text}
Step 1:"""

    @overrides()
    def few_shots_ToT(self, text):
        return f"""Explore multiple thoughts before answering:
Q: A girl ran out of the house without her bag. What might have happened?
Thought 1: She forgot the bag  
Thought 2: Something urgent made her run  
Thought 3: She was late  
Final Answer: Something urgent occurred

Q: {text}
Thought 1:
Thought 2:
Thought 3:
Final Answer:"""

    @overrides()
    def few_shots_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples)
        best_answer, all_votes = self.functions.majority_vote(samples)

        if do_print:
            st.markdown("### 🧠 Self-consistency Samples")
            st.code("\n".join(samples), language="text")
            st.markdown("**Best Answer:**")
            st.code(best_answer)
            st.markdown("**All Votes:**")
            st.code(str(all_votes))

        return best_answer, all_votes

    def select_best_path(self, thoughts, text, do_print):
        prompt = f"""You're a reasoning expert. Given a question and reasoning options, choose the best one.
Question: {text}
Options:
{chr(10).join([f"{i+1}. {t}" for i, t in enumerate(thoughts)])}
Final Answer:"""
        if do_print:
            st.markdown("### 🧠 Tree-of-Thought Voting Prompt")
            st.code(prompt, language="text")

        return self.functions.generate_output(prompt)

    def recursive_expand_tree(self, prompt, depth, breadth, do_print=False):
        if depth == 0:
            return [prompt]
        expanded = self.functions.expand_thoughts(prompt, n=breadth)
        if do_print:
            st.markdown(f"### [Depth {depth}] Prompt:")
            st.code(prompt)
            st.markdown(f"### [Depth {depth}] Thoughts:")
            st.code("\n".join(expanded))
        tree = []
        for t in expanded:
            tree += self.recursive_expand_tree(t, depth - 1, breadth, do_print)
        return tree

    def zero_shot_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.zero_shot_ToT(text)
        tree = self.recursive_expand_tree(root_prompt, depth, breadth, do_print)
        return self.select_best_path(tree, text, do_print)

    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.few_shots_ToT(text)
        tree = self.recursive_expand_tree(root_prompt, depth, breadth, do_print)
        return self.select_best_path(tree, text, do_print)

    def build_prompt(self, examples, query):
        prompt = ""
        for ex in examples:
            prompt += f"Task: Reasoning\nInput: {ex['input']}\nOutput: {ex['output']}\n"
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
