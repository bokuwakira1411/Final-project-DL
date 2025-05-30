from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
import streamlit as st

class Classification(Pattern):
    def __init__(self, model, comp_model=None, X=None, vectorizer=None, task_lib=None):
        self.functions = Global_Function(model, comp_model, X, vectorizer, task_lib)

    @overrides()
    def zero_shot_direct(self, text):
        return f"Classify the sentiment (positive, negative, or neutral) of the following sentence:\n{text}"

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""Analyze the sentence step by step, then classify the sentiment (positive, negative, neutral).
Sentence: {text}
Reasoning:"""

    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.zero_shot_CoT(text)
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
        return f"""Consider the sentiment of this sentence from three perspectives, then conclude.
Sentence: {text}
Thought 1:
Thought 2:
Thought 3:
Final Answer:"""

    @overrides()
    def few_shots_direct(self, text):
        return f"""Classify the sentiment of the sentence as positive, negative, or neutral.
          Example: "I absolutely love this!" → positive
          Example: "This is terrible." → negative
          Example: "It's fine, I guess." → neutral
          Now classify:
          "{text}" →"""

    @overrides()
    def few_shots_CoT(self, text):
        return f"""Analyze sentiment step by step:
        Example: "The service was slow, but the food was great."
        Step 1: Mixed sentiments → food positive, service negative  
        Step 2: Overall tone is neutral  
        → Answer: neutral

        Now analyze:
        "{text}"
        Step 1:"""
    
    @overrides()
    def few_shots_ToT(self, text):
        return f"""You are a sentiment analysis assistant. For each sentence, explain your reasoning and then classify it as **positive**, **negative**, or **neutral**.
            
            Example 1:
            Sentence: "The movie was boring but had a good ending."
            Reasoning: Boring → negative, Good ending → positive. The ending left a stronger impression → Sentiment: positive
            
            Example 2:
            Sentence: "I expected more, but I guess it’s fine."
            Reasoning: Disappointment shows unmet expectations, but “it’s fine” shows acceptance. Mixed feelings → Sentiment: neutral
            
            Example 3:
            Sentence: "The product is overpriced and doesn’t work well."
            Reasoning: Overpriced and malfunctioning product indicates frustration → Sentiment: negative
            
            Now analyze the following sentence:
            Sentence: "{text}"
            Reasoning:"""


    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
            root_prompt = f"""You are a sentiment analysis assistant. For each sentence, explain your reasoning and then classify it as **positive**, **negative**, or **neutral**.
            
            Example 1:
            Sentence: "The movie was boring but had a good ending."
            Reasoning: Boring → negative, Good ending → positive. The ending left a stronger impression → Sentiment: positive
            
            Example 2:
            Sentence: "I expected more, but I guess it’s fine."
            Reasoning: Disappointment shows unmet expectations, but “it’s fine” shows acceptance. Mixed feelings → Sentiment: neutral
            
            Example 3:
            Sentence: "The product is overpriced and doesn’t work well."
            Reasoning: Overpriced and malfunctioning product indicates frustration → Sentiment: negative
            
            Now analyze the following sentence:
            Sentence: "{text}"
            Reasoning:"""

            tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
            best_path = self.select_best_path(tree, text, do_print)
            return best_path
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

    def build_prompt(self, examples, query):
        prompt = ""
        for ex in examples:
            prompt += f"Task: Sentiment Classification\nInput: {ex['input']}\nOutput: {ex['output']}\n"
        prompt += f"\nNow classify:\nInput: {query}\nOutput:"
        return prompt

    def few_shots_CoT_ART(self, text, k=3):
        examples = self.functions.find_top_k_tasks(text, k)
        return self.build_prompt(examples, text)

    def run(self, text, do_print=False, type='Direct zero-shot', num_samples=5, max_len=50, k=3):
        if type == 'Zero-shot CoT + Self-consistency':
            return self.zero_shot_CoT_SC(text, num_samples, max_len, do_print)
        elif type == 'Few-shots CoT + Self-consistency':
            return self.few_shots_CoT_SC(text, num_samples, max_len, do_print)
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
            elif type == 'Few-shots CoT + ART':
                prompt = self.few_shots_CoT_ART(text, k)

            if do_print:
                st.markdown("### Prompt:")
                st.code(prompt, language="text")

            return self.functions.generate_output(prompt)
