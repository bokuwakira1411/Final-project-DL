
from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
import streamlit as st

class Classification(Pattern):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model)

    @overrides()
    def zero_shot_direct(self, text):
        return f"""
                  Instruct: Classify the sentiment (neutral, positive, negative). Don't worry about the sentence format. Sentence: {text}
                  Output: Sentiment: 
                  """

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""
                    Instruct: Classify the sentiment of this sentence as one of: neutral, positive, or negative. Sentence: "{text}
            
                    Answer: Let's think step by step. Sentiment:"""

    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.zero_shot_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        best_answer, all_votes = self.functions.majority_vote(samples)
        if do_print:
            st.markdown("### Answers (Self-consistency)")

            st.markdown("**All Sampled Answers:**")
            st.code("\n".join(samples), language="text")

            st.markdown("**Self-consistent Answer:**")
            st.code(best_answer, language="text")

            st.markdown("**All Votes:**")
            st.code(str(all_votes), language="text")
        return best_answer, all_votes

    @overrides()
    def zero_shot_ToT(self, text):
        return f"""Instruct: You are a classifier assistant. Classify the sentiment of the following sentence by considering multiple possibilities (positive, negative, neutral).
                Think in multiple directions and explain your reasoning before choosing the best label.
                {text}
                Final answer:"""

    def select_best_path(self, thoughts, text, do_print):
        prompt = f"""
        You are an expert sentiment reasoner. Based on the reasoning options below, choose the most correct label (positive, negative, or neutral) and explain your choice.
        Context: {text}
        Reasoning options:
        {chr(10).join([
            f"{i+1}. {t}" for i, t in enumerate(thoughts)
        ])}
        Evaluate each option. The most possible sentiment is (negative, positive, neutral)
        """
        if do_print:
            st.markdown("### Prompt:")
            st.code(prompt, language="text")

        input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(
            **input,
            max_length = 200,
            do_sample=False
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # def recursive_expand_tree(self, prompt, depth, breadth, do_print=False):
    #     if depth == 0:
    #         return [prompt]
    #     expanded = self.functions.expand_thoughts(prompt, n=breadth)
    #
    #     if do_print:
    #         st.markdown(f"### Prompt at Depth {depth}")
    #         st.code(prompt, language="text")
    #
    #         st.markdown(f"### Thoughts at Depth {depth}")
    #         st.code(expanded, language="text")
    #
    #
    #     tree = []
    #     for thought in expanded:
    #         sub_tree = self.recursive_expand_tree(thought, depth - 1, breadth, do_print)
    #         tree.extend(sub_tree)
    #
    #     return tree

    # def zero_shot_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
    #     root_prompt = (
    #         f"You are a sentiment classifier assistant. Use multiple reasoning paths to determine the sentiment of the sentence: {text}. "
    #         f"Consider emotional tone, polarity, and contrast. Analyze step by step, then select the most reasonable label (positive, negative, or neutral)."
    #     )
    #
    #     tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
    #     best_path = self.select_best_path(tree, text, do_print)
    #     return best_path

    @overrides()
    def few_shots_direct(self, text):
        return f"""Instruction: Classify the sentiment of each sentence as positive, neutral, or negative
                    Sentence: I love this! → positive
                    Sentence: This is terrible. → negative
                    Sentence: It works as expected. → neutral
                    Sentence: {text} 
                   Output: (only return 1 sentiment)"""

    @overrides()
    def few_shots_CoT(self, text):
        return f"""
    Instruct: Classify the sentiment of each sentence as positive, neutral, or negative.
    Sentence: I love this! → Reason: Strong positive words → Answer: positive
    Sentence: This is terrible. → Reason: Strong negative tone → Answer: negative
    Sentence: It works as expected. → Reason: Neutral tone and wording → Answer: neutral
    Sentence: {text} 
    Output: Reason: Let's think step by step. Sentiment: Final answer: """

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
        return f"""Instruct: For each sentence, explain your reasoning and then classify it as **positive**, **negative**, or **neutral**.
            
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
            Sentence: {text}
            Output: Reasoning: Final answer:"""


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
            Sentence: {text}
            Reasoning:"""

            tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
            best_path = self.select_best_path(tree, text, do_print)
            return best_path

    def run(self, text, do_print=False, type='Direct zero-shot', num_samples=5, max_len=50, depth=2, breadth=3):
        prompt = None
        max_len = 20
        if type == 'Zero-shot CoT + Self-consistency':
            return self.zero_shot_CoT_SC(text, num_samples, max_len, do_print)
            max_len = 50
        elif type == 'Few-shots CoT + Self-consistency':
            return self.few_shots_CoT_SC(text, num_samples, max_len, do_print)
        elif type == 'Zero-shot ToT expanded':
            return self.zero_shot_ToT_expanded(text, depth, breadth, do_print)
        elif type == 'Few-shots ToT expanded':
            return self.few_shots_ToT_expanded(text, depth, breadth, do_print)
        else:
            if type == 'Direct zero-shot':
                prompt = self.zero_shot_direct(text)
                max_len = 20
            elif type == 'Zero-shot CoT':
                prompt = self.zero_shot_CoT(text)
                max_len = 100
            elif type == 'Zero-shot ToT':
                prompt = self.zero_shot_ToT(text)
                max_len = 50
            elif type == 'Direct few-shots':
                prompt = self.few_shots_direct(text)
                max_len = 50
            elif type == 'Few-shots CoT':
                prompt = self.few_shots_CoT(text)
                max_len = 100
            elif type == 'Few-shots ToT':
                prompt = self.few_shots_ToT(text)
                max_len = 50
            if do_print:
                    st.markdown("### Prompt:")
                    st.code(prompt, language="text")

            input = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=False).to('cuda')
            input_len = input['input_ids'].shape[1]
            output = self.functions.generate_output(type=None, input=input, max_len=max_len)
            generate_out = output[0][input_len:]
            return self.tokenizer.decode(generate_out, skip_special_tokens=True)
