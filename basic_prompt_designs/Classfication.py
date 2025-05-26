from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function


class Classification(Pattern):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model)

    @overrides()
    def zero_shot_direct(self, text):
        return f"""
        You are a classifier assistant. Classify the sentiment of this sentence into 3 labels: negative, positive and neutral
        Q: {text}. A: The answer is ?
        """

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""
        You are a classifier assistant. Classify the sentiment of this sentence into 3 labels: negative, positive and neutral
        Q: {text}. A: Let's think step by step. The answer is ?
        """

    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.zero_shot_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        best_answer, all_votes = self.functions.majority_vote(samples)
        if do_print:
            print('Self-consistent answer: ', best_answer)
            print('All votes: ', all_votes)
        return best_answer, all_votes

    @overrides()
    def zero_shot_ToT(self, text):
        return f"""
        You are a reasoning classifier assistant. Consider multiple possibilities, classify label for each path (negative, positive, neutral)
         and choose the most reasonable one.
        Q: {text}
        Option 1: Analyze based on emotional keywords.
        Option 2: Consider the context and tone.
        Option 3: Check for contrast or negation.
        Evaluate each option and choose the best.
        Final Answer: ?
        """
    def select_best_path(self, thoughts, text, do_print):
        prompt = f"""
        You are an expert reasoner. Given the following options for reasoning, choose the most logical and correct one.
        Context: {text}
        Options:
        {chr(10).join([
            f"{i+1}. {t}" for i, t in enumerate(thoughts)
        ])}
        Answer: The best possible label is (please write clearly: positive, negative or neutral)?
        """
        if do_print:
            print('Prompt', prompt)
        input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(
            **input,
            max_length = 10,
            do_sample=False
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

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
        root_prompt = f"You are a classifier assistant. Analyze sentiment of and explain why clearly: {text}"
        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
        best_path = self.select_best_path(tree, text, do_print)
        return best_path

    @overrides()
    def few_shots_direct(self, text):
        return f"""
        You're a sentiment classifier assistant. Classify the sentiment of this sentence into 3 labels: negative, positive and neutral,
        Recorrect each time you classifying with other LLM model
          Example:Although I had high hopes, this product was a huge disappointment.
          Let's think step by step.
          1. The person had high hopes.
          2. But the product was a disappointment.
          3. The overall sentiment is negative.
          Q: {text} A: The sentiment is ?
        """

    @overrides()
    def few_shots_CoT(self, text):
        return f"""
        You're a sentiment classifier assistant. Classify the sentiment of this sentence into 3 labels: negative, positive and neutral, 
        Recorrect each time you classifying with other LLM model
          Example:Although I had high hopes, this product was a huge disappointment.
          Let's think step by step.
          1. The person had high hopes.
          2. But the product was a disappointment.
          3. The overall sentiment is negative.
          Q: {text} A: Let's think step by step. The sentiment is ?
        """

    @overrides()
    def few_shots_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        best_answer, all_votes = self.functions.majority_vote(samples)
        if do_print:
            print('Answers: ', samples)
            print('Self-consistent answer: ', best_answer)
            print('All votes: ', all_votes)
        return best_answer, all_votes

    @overrides()
    def few_shots_ToT(self, text):
        return f"""
        You're a sentiment classifier assistant. Use multiple lines of reasoning before answering.
        Example: "This movie was slow, but the ending was fantastic."
        Option 1: Slow pacing → possibly negative.
        Option 2: Fantastic ending → possibly positive.
        Option 3: Mixed emotions → possibly neutral.
        Best reasoning: Ending leaves stronger impression → Positive.
        Q: {text}
        Option 1: ...
        Option 2: ...
        Option 3: ...
        Best reasoning: ...
        Final Answer: ?
        """
    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
            root_prompt = (f"""
                        You're a sentiment classifier assistant. Use multiple lines of reasoning before answering.
                        Example: "This movie was slow, but the ending was fantastic."
                        Option 1: Slow pacing → possibly negative.
                        Option 2: Fantastic ending → possibly positive.
                        Option 3: Mixed emotions → possibly neutral.
                        Best reasoning: Ending leaves stronger impression → Positive.
                        Analyze sentiment of and explain why clearly: {text}""")
            tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
            best_path = self.select_best_path(tree, text, do_print)
            return best_path

    def run(self, text, do_print=False, type='Direct zero-shot', num_samples=5, max_len=50, depth=2, breadth=3):
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
            if do_print:
                print(prompt)
            input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
            output = self.functions.generate_output(type=None, input=input, max_len=100)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
