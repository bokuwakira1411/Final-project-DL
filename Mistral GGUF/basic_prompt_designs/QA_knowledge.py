from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
import streamlit as st

class QA_knowledge(Pattern):
    def __init__(self, model, comp_model=None, X=None, vectorizer=None, task_lib=None):
        self.functions = Global_Function(model, comp_model, X, vectorizer, task_lib)

    @overrides()
    def zero_shot_direct(self, text):
        return f"Answer the following question concisely and factually:\n{text}"

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""Answer the following question step by step using your knowledge:
Question: {text}
Answer:"""

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
        return f"""Answer this question by exploring multiple knowledge-based reasoning paths:
Question: {text}
Thought 1:
Thought 2:
Thought 3:
Final Answer:"""

    @overrides()
    def few_shots_direct(self, text):
        return f"""Answer the following factual questions directly:
Q: Who wrote 'Pride and Prejudice'?
A: Jane Austen
Q: {text}
A:"""

    @overrides()
    def few_shots_CoT(self, text):
        return f"""Use step-by-step reasoning to answer:
Q: Why does the moon affect the tides?
Step 1: The moon's gravity pulls on Earth's oceans.  
Step 2: This pull creates a bulge in water levels on the side facing the moon.  
→ Answer: The gravitational pull of the moon causes tidal forces.

Now try:
Q: {text}
Step 1:"""

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
        return f"""Explore multiple perspectives before answering:
Q: Why is the sky blue?
Thought 1: Sunlight hits molecules and scatters — shorter blue wavelengths scatter more.  
Thought 2: Human eyes are more sensitive to blue light.  
Thought 3: Blue light reaches us from all directions.  
Final Answer: Due to Rayleigh scattering, blue light dominates.

Now try:
Q: {text}
Thought 1:
Thought 2:
Thought 3:
Final Answer:"""

    def build_prompt(self, examples, query):
        prompt = ""
        for ex in examples:
            prompt += f"Task: QA\nInput: {ex['input']}\nOutput: {ex['output']}\n"
        prompt += f"\nNow answer this:\nInput: {query}\nOutput:"
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
            elif type == 'Few-shots ToT':
                prompt = self.few_shots_ToT(text)
            elif type == 'Few-shots CoT + ART':
                prompt = self.few_shots_CoT_ART(text, k)

            if do_print:
                st.markdown("### Prompt:")
                st.code(prompt, language="text")

            return self.functions.generate_output(prompt)
