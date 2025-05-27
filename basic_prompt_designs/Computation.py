from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
from sentence_transformers import SentenceTransformer, util


class Computation(Pattern):
    def __init__(self, tokenizer, model, comp_model, X, vectorizer, task_lib):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model, comp_model, X, vectorizer, task_lib)

    @overrides()
    def zero_shot_direct(self, text):
        return f"""
        You are a math assistant. Solve this question by using math formulas
        Q: {text}. A:You need to answer follow this form: The answer is ?
        """

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""
        You are a math assistant. Solve this question by using math formulas
        Q: {text}. A: Let's think step by step. You need to answer follow this form: The answer is ?
        """

    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        best_answer, all_votes = self.functions.majority_vote(samples)
        if do_print:
            print('Answers: ', samples)
            print('Self-consistent answer: ', best_answer)
            print('All votes: ', all_votes)
        return best_answer, all_votes

    @overrides()
    def zero_shot_ToT(self, text):
        return f"""
        You are a math assistant. Solve this question by identifying multiple thoughts or approaches.
        Q: {text} A: Let's try solving this from different reasoning paths. The answer is ?
        """

    def select_best_path(self, thoughts, text, do_print):
        prompt = f"""
                You are a math expert. Given a question and several possible reasoning paths, select the best and most logically sound one.

                Context:
                {text}

                Computing Options:
                {chr(10).join([f"{i + 1}. {t}" for i, t in enumerate(thoughts)])}

                Please reply with the number of the best reasoning option and explain briefly why it is the best.
                The final answer is:
                """
        if do_print:
            print('Prompt:\n', prompt)

        input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(
            **input,
            max_length=300,
            do_sample=False
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return response

    def zero_shot_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = f"You are a math assistant. Explore different possible  Solve this question by identifying multiple thoughts or approaches and express each as a distinct reasoning path:\n\n{text}"
        if do_print:
            print('Root_prompt:\n', root_prompt)

        thoughts = self.functions.expand_thoughts(root_prompt, n=breadth)
        if do_print:
            print('Initial thoughts:', thoughts)

        tree = []
        for t in thoughts:
            sub_thoughts = self.functions.expand_thoughts(t, n=breadth)
            tree.append(sub_thoughts)

        # Flatten the tree into a single list of candidate answers
        candidates = [s for group in tree for s in group]
        best_path = self.select_best_path(candidates, text, do_print)

        return best_path

    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.few_shots_ToT(text)

        thoughts = self.functions.expand_thoughts(root_prompt, n=breadth)
        if do_print:
            print('Root_prompt:\n', root_prompt)
            print('Thoughts:', thoughts)

        tree = []
        for t in thoughts:
            sub_thoughts = self.functions.expand_thoughts(t, n=breadth)
            tree.append(sub_thoughts)

        # Flatten tree again
        candidates = [s for group in tree for s in group]
        best_path = self.select_best_path(candidates, text, do_print)

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
        Q: {text} A: The answer is ?
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
        Q: {text} A: Let's think step by step. The answer is ?
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
        The answer is?
        """

    def build_prompt(self, examples, query):
        prompt = ""
        for ex in examples:
            prompt += f"""Task: {ex.get('instruction')}
            Input: {ex['input']}
            Output: {ex['output']}
            """
        prompt += (
            "Now solve this task:\n"
            f"Input: {query}\n"
            "Solution:\n"
            "Let's solve this step-by-step. Write your reasoning clearly."
        )
        return prompt

    def few_shots_CoT_ART(self, text):
        examples = self.functions.find_top_k_tasks(text, 3)
        return self.build_prompt(examples, text)


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
            output = self.functions.generate_output(type='generation', input=input, max_len=300)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
