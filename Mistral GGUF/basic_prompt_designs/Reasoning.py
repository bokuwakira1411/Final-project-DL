from overrides import overrides
from basic_prompt_designs.Pattern import Pattern
from basic_prompt_designs.Global_Function import Global_Function
from sentence_transformers import SentenceTransformer, util
import re
import streamlit as st

class Reasoning(Pattern):
    def __init__(self, tokenizer, model, comp_model, X, vectorizer, task_lib):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model, comp_model, X, vectorizer, task_lib)

    def get_best_thought(self, thoughts, text, do_print=False):
        output = self.select_best_path(thoughts, text, do_print=do_print)
        if 0 < len(output) < 3:
            match = re.search(r"\b(\d+)\b", output)
            if match:
                index = int(match.group(1)) - 1
                if 0 <= index < len(thoughts):
                    return thoughts[index]
        else:
            return output

    @overrides()
    def zero_shot_direct(self, text):
        return f"""Instruct: Read the article and answer the question based only on the information provided. Then explain your reasoning in 1-2 short sentences.
                {text}
            
                Answer:"""

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""Instruct: Read the article and answer the question based only on the information provided. Then explain your reasoning in 1-2 short sentences.
                   {text}. "
                   Answer: Let's think step by step"""

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
        Instruct: Break down the problem by exploring multiple lines of reasoning. Each thought should follow a different logical path.
        {text}
        Thought 1: Approach the question using common sense reasoning.
        Thought 2: Consider possible assumptions and implications.
        Thought 3: Analyze the question from a cause-effect perspective.
        After exploring all thoughts, decide on the most reasonable final answer.
        Answer:
        """

    @overrides()
    def few_shots_direct(self, text):
        return f"""
    Instruct: Read the question and give a direct but well-thought-out answer, without listing multiple thoughts. Here are some examples:
    Question: Phytochemistry is a branch of plant biochemistry primarily concerned with the chemical substances produced by plants during secondary metabolism. Some of these compounds are toxins such as the alkaloid coniine from hemlock. Others, such as the essential oils peppermint oil and lemon oil are useful for their aroma, as flavourings and spices (e.g., capsaicin), and in medicine as pharmaceuticals as in opium from opium poppies. Many medicinal and recreational drugs, such as tetrahydrocannabinol (active ingredient in cannabis), caffeine, morphine and nicotine come directly from plants. Others are simple derivatives of botanical natural products. For example, the pain killer aspirin is the acetyl ester of salicylic acid, originally isolated from the bark of willow trees, and a wide range of opiate painkillers like heroin are obtained by chemical modification of morphine obtained from the opium poppy. Popular stimulants come from plants, such as caffeine from coffee, tea and chocolate, and nicotine from tobacco. Most alcoholic beverages come from fermentation of carbohydrate-rich plant products such as barley (beer), rice (sake) and grapes (wine).\n\nNow answer this question: Where do some medicines and recreational drugs come from?    A: Someone accidentally hit the ball through the window.
    Answer: from plants.
    Explanation: The article states that many medicinal and recreational drugs, such as tetrahydrocannabinol (active ingredient in cannabis), caffeine, morphine and nicotine come directly from plants. These are some examples of the medicines found in plants mentioned by the author. Thus it can be stated with certainty that some medicines do indeed come from plants.\n\nTherefore, \"from plants\" is the correct answer option to this question based on the context provided.
    Question: In this task, you are provided with an article of the legal acts. Your task is to classify it into three categories (Regulation, Decision and Directive) based on its content: 1) Regulation is a binding legislative act that must be applied in its entirety on a set date across all the member states (European Union countries). 2) Decision is binding on those to whom it is addressed (e.g. an European Union country or an individual company) and is directly applicable. 3) Directive is a legislative act that sets out a goal that all  must achieve. However, it is up to the individual countries to devise their own laws on how to reach these goals.\n\nProof that the special export tax mentioned in Articles 2 and 3 of Regulation (EEC) No 1234\/71 has been paid shall be furnished to the competent authority of the importing Member State by presentation of movement certificate A.TR.1. In that case, one of the following entries shall be made in the 'Remarks' section by the competent authority:'Taxe spéciale à l exportation selon règlement (CEE) No 1234\/71 acquittée pour un montant de ...''Besondere Ausfuhrabgabe gemäss Verordnung (EWG) nr. 1234\/71 in Höhe von ... entrichtet.''Tassa speciale per l esportazione pagata, secondo regolamento (CEE) n 1234\/71, per un importo di ...''Speciale heffing bij uitvoer bedoeld in Verordening (EEG) nr 1234\/71 ten bedrage van ... voldaan'.Special export tax in accordance with Regulation (EEC) No 1234\/71 paid in the amount of ... . Commission Regulation (EEC) No 2019\/71 of 20 September 1971 is hereby repealed.This Regulation shall enter into force on the third day following its publication in the Official Journal of the European Communities.This Regulation shall be binding in its entirety and directly applicable in all Member States. 
    Answer: Regulation
    Explanation: This act has all the features of Regulation. The act is addressed to an European Union country; it must be applied once and for all within a certain time-limit; it lays down general rules of application, which are binding on all Member States.
    Question: What's the answer to that question: where did lee surrender to grant to end the civil war? Choices: ",
    Answer: Battle of Appomattox Court House
    Explanation: The Battle of Appomattox Court House was a battle in the final stages of the American Civil War, resulting in Confederate General Robert E. Lee surrendering his Army to Union Commander Ulysses S. Grant on April 9th 1865"
    Now solve this question
    Question: {text}
    Answer:
    Explanation:
    """

    @overrides()
    def few_shots_ToT(self, text):
        return f"""
You are a reasoning assistant. For each question, explore multiple lines of reasoning. Each thought should follow a different logical path. You must explain clearly in each path. Only after that, you give the final answer.

Q: Where do some medicines and recreational drugs come from?
Context: Phytochemistry is a branch of plant biochemistry primarily concerned with the chemical substances produced by plants during secondary metabolism. [...] Popular stimulants come from plants, such as caffeine from coffee, tea and chocolate, and nicotine from tobacco.
Choices:
A. from animals
B. from plants
C. synthetically only
D. from minerals

Thought 1 (Common sense reasoning):
We commonly know that many drugs like caffeine, nicotine, and morphine are derived from natural sources like tea, tobacco, and poppy plants. These are familiar examples that most people recognize as plant-derived substances.

Thought 2 (Assumptions and implications):
The question implies a general source for medicines and recreational drugs. The context gives multiple examples of plant-based compounds, and no examples from animals or synthetic-only processes. So it's implied that plants are a primary source. If the answer were "animals" or "synthetic only," the text would need to mention them, but it does not.

Thought 3 (Cause-effect analysis):
Plants undergo secondary metabolism which produces chemical compounds like alkaloids. These have strong physiological effects on humans (e.g., caffeine stimulates the nervous system). That biological capability explains why plants are a common origin of such drugs.

Final Answer: from plants

---

Q: Why do metal objects feel colder than wooden ones at the same temperature?
Context: Even when objects are at the same temperature, materials conduct heat differently. Metals are good conductors of heat, while wood is a poor conductor.
Choices:
A. Because metals are colder
B. Because wood generates heat
C. Because metals conduct heat away from the skin faster
D. Because wood holds heat longer

Thought 1 (Common sense reasoning):
Touching a metal door handle feels much colder than touching a wooden table, even if both have been in the same room. This everyday experience suggests metal pulls heat away from us more quickly.

Thought 2 (Assumptions and implications):
The assumption is that both materials are at the same temperature, so the cause of the sensation must be related to heat transfer, not temperature. The implication is that perception of "coldness" comes from heat leaving our body faster when we touch metal.

Thought 3 (Cause-effect analysis):
Metals are thermal conductors — they rapidly transfer heat from our warm skin into the cooler material. Wood is an insulator, so the heat stays near the contact area and doesn't flow away as fast, making it feel warmer to touch.

Final Answer: Because metals conduct heat away from the skin faster

---

Q: {text}
Thought 1 (Common sense reasoning):
Thought 2 (Assumptions and implications):
Thought 3 (Cause-effect analysis):

Final Answer:
"""

    def select_best_path(self, thoughts, text, do_print=False):
        prompt = f"""You are a reasoning expert. Given a question and several reasoning paths, choose the most logical one.

        Question: {text}

        Candidates:
        {chr(10).join([f"{i + 1}. {t}" for i, t in enumerate(thoughts)])}

        Reply with the best option, justify briefly. Final answer is ?"""
        if do_print:
            print('prompt:\n', prompt)
        print(thoughts)
        input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        input_len = input['input_ids'].shape[1]
        output = self.functions.generate_output(type=None, input=input)[0]
        generate_ids = output[0][input_len:]
        answer = self.tokenizer.decode(generate_ids, skip_special_tokens=True).strip()
        return answer

    def expand_thoughts(self, prompt, n=3):
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        outputs = [self.model.generate(**inputs, max_length=150) for _ in range(n)]
        return [self.tokenizer.decode(o[0], skip_special_tokens=True) for o in outputs]

    def recursive_expand_tree(self, prompt, depth, breadth, context, do_print=False):
        if depth == 0:
            return [prompt]

        base_prompt = f"""
            You are a reasoning assistant. Use tree-of-thought reasoning to explore solutions.

            Question: {context}

            Partial Reasoning:
            {prompt}
            Now expand with next steps:
            """

        expanded = self.expand_thoughts(base_prompt, n=breadth)
        expanded = list(set(expanded))
        if do_print:
            print(f"[Depth {depth}] Base prompt:\n{base_prompt}")
            print(f"[Depth {depth}] Got thoughts:\n", expanded)

        tree = []
        for thought in expanded:
            sub_tree = self.recursive_expand_tree(thought, depth - 1, breadth, context, do_print)
            tree.extend(sub_tree)
        return tree

    def zero_shot_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.zero_shot_ToT(text)
        if do_print:
            print('Root_prompt', root_prompt)
        tree = self.recursive_expand_tree(root_prompt, depth, breadth, text, do_print)
        best = self.get_best_thought(tree, text, do_print)
        return best

    def few_shots_ToT_expanded(self, text, depth=2, breadth=3, do_print=False):
        root_prompt = self.few_shots_ToT(text)
        if do_print:
            print('Root_prompt', root_prompt)
        tree = []
        for _ in range(breadth):
            single_tree = self.recursive_expand_tree(root_prompt, depth, breadth, text, do_print)
            tree.extend(single_tree)

        best_path = self.get_best_thought(tree, text, do_print)
        return best_path
    @overrides()
    def few_shots_CoT(self, text):
        return f"""
                Instruct: Read the question and give a direct but well-thought-out answer, without listing multiple thoughts. Here are some examples:
                Question: Phytochemistry is a branch of plant biochemistry primarily concerned with the chemical substances produced by plants during secondary metabolism. Some of these compounds are toxins such as the alkaloid coniine from hemlock. Others, such as the essential oils peppermint oil and lemon oil are useful for their aroma, as flavourings and spices (e.g., capsaicin), and in medicine as pharmaceuticals as in opium from opium poppies. Many medicinal and recreational drugs, such as tetrahydrocannabinol (active ingredient in cannabis), caffeine, morphine and nicotine come directly from plants. Others are simple derivatives of botanical natural products. For example, the pain killer aspirin is the acetyl ester of salicylic acid, originally isolated from the bark of willow trees, and a wide range of opiate painkillers like heroin are obtained by chemical modification of morphine obtained from the opium poppy. Popular stimulants come from plants, such as caffeine from coffee, tea and chocolate, and nicotine from tobacco. Most alcoholic beverages come from fermentation of carbohydrate-rich plant products such as barley (beer), rice (sake) and grapes (wine).\n\nNow answer this question: Where do some medicines and recreational drugs come from?    A: Someone accidentally hit the ball through the window.
                Answer: from plants.
                Explanation: The article states that many medicinal and recreational drugs, such as tetrahydrocannabinol (active ingredient in cannabis), caffeine, morphine and nicotine come directly from plants. These are some examples of the medicines found in plants mentioned by the author. Thus it can be stated with certainty that some medicines do indeed come from plants.\n\nTherefore, \"from plants\" is the correct answer option to this question based on the context provided.
                Question: In this task, you are provided with an article of the legal acts. Your task is to classify it into three categories (Regulation, Decision and Directive) based on its content: 1) Regulation is a binding legislative act that must be applied in its entirety on a set date across all the member states (European Union countries). 2) Decision is binding on those to whom it is addressed (e.g. an European Union country or an individual company) and is directly applicable. 3) Directive is a legislative act that sets out a goal that all  must achieve. However, it is up to the individual countries to devise their own laws on how to reach these goals.\n\nProof that the special export tax mentioned in Articles 2 and 3 of Regulation (EEC) No 1234\/71 has been paid shall be furnished to the competent authority of the importing Member State by presentation of movement certificate A.TR.1. In that case, one of the following entries shall be made in the 'Remarks' section by the competent authority:'Taxe spéciale à l exportation selon règlement (CEE) No 1234\/71 acquittée pour un montant de ...''Besondere Ausfuhrabgabe gemäss Verordnung (EWG) nr. 1234\/71 in Höhe von ... entrichtet.''Tassa speciale per l esportazione pagata, secondo regolamento (CEE) n 1234\/71, per un importo di ...''Speciale heffing bij uitvoer bedoeld in Verordening (EEG) nr 1234\/71 ten bedrage van ... voldaan'.Special export tax in accordance with Regulation (EEC) No 1234\/71 paid in the amount of ... . Commission Regulation (EEC) No 2019\/71 of 20 September 1971 is hereby repealed.This Regulation shall enter into force on the third day following its publication in the Official Journal of the European Communities.This Regulation shall be binding in its entirety and directly applicable in all Member States. 
                Answer: Regulation
                Explanation: This act has all the features of Regulation. The act is addressed to an European Union country; it must be applied once and for all within a certain time-limit; it lays down general rules of application, which are binding on all Member States.
                Question: What's the answer to that question: where did lee surrender to grant to end the civil war? Choices: ",
                Answer: Battle of Appomattox Court House
                Explanation: The Battle of Appomattox Court House was a battle in the final stages of the American Civil War, resulting in Confederate General Robert E. Lee surrendering his Army to Union Commander Ulysses S. Grant on April 9th 1865"
                Now solve this question
                Question: {text}
                Answer:
                Explanation: Let's think step by step """

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

    def build_prompt(self, examples, query):
        prompt = ""
        for ex in examples:
            prompt += f"""
            Task: Reasoning for text
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

            input = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=False).to('cuda')
            input_len = input['input_ids'].shape[1]
            output = self.functions.generate_output(type=None, input=input, max_len=max_len)
            generate_out = output[0][input_len:]
            return self.tokenizer.decode(generate_out, skip_special_tokens=True)
