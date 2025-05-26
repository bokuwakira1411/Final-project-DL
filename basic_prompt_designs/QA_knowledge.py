from overrides import overrides
import Pattern
from basic_prompt_designs.Global_Function import Global_Function
from sentence_transformers import SentenceTransformer, util

class QA_knowledge(Pattern):
    def __init__(self, tokenizer, model, comp_model, X, vectorizer, task_lib):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model, comp_model, X, vectorizer, task_lib)
    @overrides()
    def zero_shot_direct(self, text):
        return f"""
        Answer the following question by your knowledge.
        Q: {text}. A: The answer is ?
        """

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""
                Answer the following question by your knowledge.
                Q: {text}. A: Let's think step by step and explain why. The answer is ?
                """

    @overrides()
    def zero_shot_ToT(self, text):
        pass
    @overrides()
    def few_shots_direct(self, text):
        pass
    @overrides()
    def few_shots_CoT(self, text):
        pass

    @overrides()
    def few_shots_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.zero_shot_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        best_answer, all_votes = self.functions.majority_vote(samples)
        if do_print:
            print('Answers: ', samples)
            print('Self-consistent answer: ', best_answer)
            print('All votes: ', all_votes)
        return best_answer, all_votes

    @overrides()
    def few_shots_ToT(self, text):
        pass

    def build_prompt(examples, query):
        prompt = ""
        for ex in examples:
            prompt += f"""Task: {ex.get('instruction')}
                        Input: {ex['input']}
                        Output:
                        {ex['output']}
                        """

        prompt += (
            "Now solve this task:\n"
            f"Input: {query}\n"
            "Solution:\n"
            "Let's solve this step-by-step. Write your reasoning clearly"
        )
        return prompt
    def few_shots_CoT_ART(self, text):
        ex = self.functions.find_top_k_tasks(text, 3)
        prompt = self.build_prompt(ex, text)
        return prompt

    def run(self, text, do_print=False, type='Direct zero-shot', type_output = None, num_samples=5, max_len=50, ):
        prompt = None
        if type == 'Zero-shot CoT + Self-consistency':
            self.zero_shot_CoT_SC(text, num_samples, max_len, do_print)
        elif type == 'Few-shots CoT + Self-consistency':
            self.few_shots_CoT_SC(text, num_samples, max_len, do_print)
        else:
            if type == 'Direct zero-shot':
                prompt = self.zero_shot_direct(text)
            elif type == 'Zero-shot CoT':
                prompt = self.zero_shot_CoT(text)
            elif type == 'Zero-shot CoT + Self-consistency':
                self.zero_shot_CoT_SC(text, num_samples, max_len, do_print)
            elif type == 'Zero-shot ToT':
                prompt = self.zero_shot_ToT(text)
            elif type == 'Direct few-shots':
                prompt = self.few_shots_direct(text)
            elif type == 'Few-shots CoT':
                prompt = self.few_shots_CoT(text)
            elif type == 'Few-shots ToT':
                prompt = self.few_shots_ToT(text)
            elif type == 'Few-shots CoT ART':
                prompt = self.few_shots_CoT_ART(text)
            input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
            output = self.functions.generate_output(type=type_output, input=input, max_len=100)
            if do_print==True:
                print(prompt)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
