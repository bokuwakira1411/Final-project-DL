from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
from sentence_transformers import SentenceTransformer, util


class Reasoning(Pattern):
    def __init__(self, tokenizer, model, comp_model, X, vectorizer, task_lib):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model, comp_model, X, vectorizer, task_lib)

    @overrides()
    def zero_shot_direct(self, text):
        return f"""
        You are a reasoning assistant. Break down the problem by exploring multiple lines of reasoning. 
        Q: {text}. A:Answer the question
        """

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""
                You are a reasoning assistant. Break down the problem by exploring multiple lines of reasoning. 
                Q: {text}. A: Let's think step by step. Answer the question
                """
    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.zero_shot_CoT(text)
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
        You are a reasoning assistant. Break down the problem by exploring multiple lines of reasoning. Each thought should follow a different logical path.

        Q: {text}

        Thought 1: Approach the question using common sense reasoning.
        Thought 2: Consider possible assumptions and implications.
        Thought 3: Analyze the question from a cause-effect perspective.

        After exploring all thoughts, decide on the most reasonable final answer.

        A:Answer the question
        """

    @overrides()
    def few_shots_direct(self, text):
        return f"""
    You are a reasoning assistant. Read the question and give a direct but well-thought-out answer, without listing multiple thoughts.

    Q: A man walks into a room and sees a broken window and a baseball lying on the floor. What probably happened?
    A: Someone accidentally hit the ball through the window.

    Q: A woman hears the fire alarm and smells smoke coming from the kitchen. What should she do?
    A: She should quickly check the kitchen and call emergency services if there's a fire.

    Q: A student studied hard for an exam and felt confident after taking it. What is the likely outcome?
    A: The student probably performed well on the exam.
    
    Q: {text}
    A:Answer the question
    """

    @overrides()
    def few_shots_ToT(self, text):
        return f"""
        You are a reasoning assistant. For each question, explore multiple reasoning paths (thoughts). After considering them, choose the most reasonable final answer.
    
        Q: A student missed school for three days. On the first day, he had a headache. On the second day, he felt better but stayed home. On the third day, he went to school. Why did he return on the third day?
        Thought 1: He returned because the headache was gone.
        Thought 2: He felt better by the second day, so the third day was safe to return.
        Thought 3: He may have had an important class or test on the third day.
        Final Answer: Thought 2 seems most consistent with the timeline and motivation. He returned because he felt well enough.
    
        Q: A man enters a bakery and buys a cake. As he leaves, he slips on the floor and drops the cake. Why might he be upset?
        Thought 1: He wasted money buying the cake and now can’t eat it.
        Thought 2: The bakery floor might have been slippery, which caused the accident.
        Thought 3: He may have bought the cake for a special occasion.
        Final Answer: Thought 3 provides deeper emotional motivation. He is upset because the cake was for a special event.
    
        Q: {text}
        Thought 1:
        Thought 2:
        Thought 3:
        A:Answer the question"""

    def select_best_path(self, thoughts, text, do_print):
        prompt = f"""
                You are a reasoning expert. Given a question and several possible reasoning paths, select the best and most logically sound one.
            
                Context:
                {text}
            
                Reasoning Options:
                {chr(10).join([f"{i + 1}. {t}" for i, t in enumerate(thoughts)])}
            
                Please reply with the number of the best reasoning option and explain briefly why it is the best.
                A:Answer the question
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
        root_prompt = f"You are a reasoning assistant. Explore different possible interpretations of the situation and express each as a distinct reasoning path:\n\n{text}"
        if do_print:
            print('Root_prompt', root_prompt)
        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
        best_path = self.select_best_path(tree, text, do_print)
        return best_path

    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = f"""
            You are a reasoning assistant. For each question, explore multiple possible reasoning paths (called thoughts), then choose the most reasonable final answer.
        
            Q: A student missed school for three days. On the first day, he had a headache. On the second day, he felt better but stayed home. On the third day, he went to school. Why did he return on the third day?
            Thought 1: He returned because the headache was gone.
            Thought 2: He felt better by the second day, so the third day was safe to return.
            Thought 3: He may have had an important class or test on the third day.
            Final Answer: Thought 2 seems most consistent with the timeline and motivation. He returned because he felt well enough.
        
            Q: A man enters a bakery and buys a cake. As he leaves, he slips on the floor and drops the cake. Why might he be upset?
            Thought 1: He wasted money buying the cake and now can’t eat it.
            Thought 2: The bakery floor might have been slippery, which caused the accident.
            Thought 3: He may have bought the cake for a special occasion.
            Final Answer: Thought 3 provides deeper emotional motivation. He is upset because the cake was for a special event.
        
            Q: {text}
            Thought 1:
            Thought 2:
            Thought 3:
            A:Answer the question
            """
        if do_print:
            print('Root_prompt', root_prompt)
        tree = self.recursive_expand_tree(root_prompt, depth=depth, breadth=breadth, do_print=do_print)
        best_path = self.select_best_path(tree, text, do_print)
        return best_path
    @overrides()
    def few_shots_CoT(self, text):
        pass
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
    def few_shots_CoT_ART(self, text):
        pass

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
            output = self.functions.generate_output(type=None, input=input, max_len=300)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
