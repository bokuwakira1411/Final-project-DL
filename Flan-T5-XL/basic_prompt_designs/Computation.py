from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
from sentence_transformers import SentenceTransformer, util

import streamlit as st
class Computation(Pattern):
    def __init__(self, tokenizer, model, comp_model, X, vectorizer, task_lib):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model, comp_model, X, vectorizer, task_lib)

    @overrides()
    def zero_shot_direct(self, text):
        return f"""
        You are a math assistant. Solve this question by using math formulas
        Q: {text}. A:You need to answer follow this form and always calculate all formulas.: The answer is ?
        """

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""
        You are a math assistant. Solve this question by using math formulas
        Q: {text}. A: Let's think step by step. You need to answer follow this form and always calculate all formulas: The answer is ?
        """

    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        best_answer, all_votes = self.functions.majority_vote(samples)
        if do_print:
            st.markdown("### 📝 Answers (Self-consistency)")

            st.markdown("**All Sampled Answers:**")
            st.code("\n".join(samples), language="text")

            st.markdown("**Self-consistent Answer:**")
            st.code(best_answer, language="text")

            st.markdown("**All Votes:**")
            st.code(str(all_votes), language="text")
        return best_answer, all_votes

    @overrides()
    def zero_shot_ToT(self, text):
        return f"""
        You are a math assistant. Solve this question by identifying multiple thoughts or approaches.
        Q: {text} A: Let's try solving this from different reasoning paths and always calculate all formulas. The answer is ?
        """

    def select_best_path(self, thoughts, text, do_print):
        prompt = f"""
                You are a math expert. Given a question and several possible reasoning paths, select the best and most logically sound one.
                Context:
                {text}
                Computing Options:
                {chr(10).join([f"{i + 1}. {t}" for i, t in enumerate(thoughts)])}
                Please reply with the number of the best reasoning option and explain briefly why it is the best.
                Always calculate all formulas.
                The final answer is:
                """
        if do_print:
            st.markdown("### Prompt:")
            st.code(prompt, language="text")
        input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(
            **input,
            max_length=300,
            do_sample=False
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return response

    def recursive_expand_tree(self, prompt, depth, breadth, do_print=False):
        if depth == 0:
            return [prompt]
        expanded = self.functions.expand_thoughts(prompt, n=breadth)
        if do_print:
            print(f"[Depth {depth}] Expand prompt:\n{prompt}")
            print(f"[Depth {depth}] Got thoughts:\n", expanded)
        tree = []
        for thought in expanded:
            sub_tree = self.recursive_expand_tree(thought, depth - 1, breadth, do_print)
            tree.extend(sub_tree)
        return tree

    def zero_shot_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.zero_shot_ToT(text)
        thoughts = self.functions.expand_thoughts(root_prompt, n=breadth)
        if do_print:
            st.markdown("### Thoughts:")
            st.code(thoughts, language="text")
        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
        best_path = self.select_best_path(tree, text, do_print)
        return best_path

    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.few_shots_ToT(text)
        thoughts = self.functions.expand_thoughts(root_prompt, n=breadth)
        if do_print:
            print('Root_prompt:\n', root_prompt)
            print('Thoughts:', thoughts)
        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
        best_path = self.select_best_path(tree, text, do_print)
        return best_path

    @overrides()
    def few_shots_direct(self, text):
        return f"""
        You are a math assistant. Whenever you see a math expression, call the function CALC().
        Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        A: Natalia sold CALC(48/2)=24 clips in May.
        Natalia sold CALC(48+24)=72 clips altogether in April and May.
        #### 72
        Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
        A: Weng earns $CALC(12/60)=0.2 per minute.
        Working 50 minutes, she earned $CALC(0.2*50)=10,
        #### 10
        Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
        A: In the beginning, Betty has only $CALC(100/2) = 50.
        Betty's grandparents gave her $CALC(15*2)=30.
        This means, Betty needs $CALC(100-50-30-15) = 5 more.
        #### 5
        Q: {text} A: Always calculate all formulas. The answer is ?
        """

    @overrides()
    def few_shots_CoT(self, text):
        return f"""
        You are a math assistant. Whenever you see a math expression, call the function CALC().
        Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        A: Natalia sold CALC(48/2)=24 clips in May.
        Natalia sold CALC(48+24)=72 clips altogether in April and May.
        #### 72
        Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
        A: Weng earns $CALC(12/60)=0.2 per minute.
        Working 50 minutes, she earned $CALC(0.2*50)=10,
        #### 10
        Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
        A: In the beginning, Betty has only $CALC(100/2) = 50.
        Betty's grandparents gave her $CALC(15*2)=30.
        This means, Betty needs $CALC(100-50-30-15) = 5 more.
        #### 5
        Q: {text} A: Let's think step by step and always calculate all formulas. The answer is ?
        """

    @overrides()
    def few_shots_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        best_answer, all_votes = self.functions.majority_vote(samples)
        if do_print:
            st.markdown("### 📝 Answers (Self-consistency)")

            st.markdown("**All Sampled Answers:**")
            st.code("\n".join(samples), language="text")

            st.markdown("**Self-consistent Answer:**")
            st.code(best_answer, language="text")

            st.markdown("**All Votes:**")
            st.code(str(all_votes), language="text")
        return best_answer, all_votes

    @overrides()
    def few_shots_ToT(self, text):
        return f"""
        You are a math assistant.\n
         Use a tree-of-thought approach to break down complex problems by exploring different solution paths, then converge on the correct answer.

        Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        A:
        Thought 1: Natalia sold 48 clips in April.
        Thought 2: She sold half as many in May → CALC(48/2)=24 clips.
        Thought 3: Total clips = CALC(48+24)=72.
        The final answer is 72

        Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
        A:
        Thought 1: Convert hourly rate to per-minute → CALC(12/60)=0.2.
        Thought 2: Multiply by minutes worked → CALC(0.2*50)=10.
        The final answer is 10

        Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
        A:
        Thought 1: Betty initially has CALC(100/2)=50.
        Thought 2: Parents give her $15, grandparents give CALC(15*2)=30.
        Thought 3: Total amount Betty has = CALC(50+15+30)=95.
        Thought 4: She still needs CALC(100-95)=5 more.
        The final answer is 5

        Q: {text}
        A:
        Let's think in steps, exploring each possibility:
        Thought 1:
        Thought 2:
        Thought 3:
        Always calculate all formulas.The answer is?
        """

    def build_prompt(self, examples, query):
        prompt = ""
        for ex in examples:
            prompt += f"""
            Task: Mathematics Inference
            Input: {ex['input']}
            Output: {ex['output']}
            """
            prompt += (
                "Now solve this task:\n"
                f"Input: {query}\n"
                "Solution:\n"
                "Let's solve this step-by-step. Always calculate all formulas and write your reasoning clearly."
            )
        return prompt

    def few_shots_CoT_ART(self, text, k=3):
        examples = self.functions.find_top_k_tasks(text, k)
        return self.build_prompt(examples, text)


    def run(self, text, do_print=False, type='Direct zero-shot', num_samples=5, max_len=50, depth=2, breadth=3, k=3):
        prompt = None
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
            input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
            output = self.functions.generate_output(type='generation', input=input, max_len=300)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
