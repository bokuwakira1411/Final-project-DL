from overrides import overrides
from basic_prompt_designs.Global_Function import Global_Function
from basic_prompt_designs.Pattern import Pattern
from sentence_transformers import SentenceTransformer, util
import streamlit as st
class QA_knowledge(Pattern):
    def __init__(self, tokenizer, model, comp_model, X, vectorizer, task_lib):
        self.tokenizer = tokenizer
        self.model = model
        self.functions = Global_Function(tokenizer, model, comp_model, X, vectorizer, task_lib)
    @overrides()
    def zero_shot_direct(self, text):
        return f"""Instruction: Answer clearly, exactly the following :\n{text}. You can use reliable sources to answer such as google, wikipedia,....\n
                 Answer: """

    @overrides()
    def zero_shot_CoT(self, text):
        return f"""Instruction:  {text}\n
                   Answer: Let's break it down and then answer clearly, exactly the following."""

    @overrides()
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.zero_shot_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        if do_print:
            st.code(prompt, language='text')

        return samples

    @overrides()
    def few_shots_direct(self, text):
        return f"""
            Instruction: Answer the following question, using reliable sources such as wikipedia,...Here are some examples:
            Question:between 1900 and 1920 where did most of the migrants to the united states come from,
            Answer: During the period between 1900 and 1920, most migrants to the United States came from Europe, particularly from Italy, Poland, and Russia. There were also significant numbers of immigrants from Canada, Mexico, and other parts of Latin America. In addition, there were smaller numbers of immigrants from Asia and other parts of the world.

            Question: What does inioluwa mean,
            Answer: Inioluwa is a Nigerian name that means \"God's gift\" in Yoruba, a language spoken in Nigeria. It is typically given to a child who is seen as a blessing from God.

            Question: what is CVD,
            Answer: CVD stands for Cardiovascular Disease. It is a group of conditions that affect the heart and blood vessels. These conditions include coronary artery disease (which can lead to heart attacks),
            
            Now answer clearly, exactly the following: {text}, do not offer irrelevant information.
            Answer:\n 

                """
    @overrides()
    def few_shots_CoT(self, text):
        return f"""Instruction: Answer the question using reliable information. Think step by step.

                Question:between 1900 and 1920 where did most of the migrants to the united states come from,
                Answer: During the period between 1900 and 1920, most migrants to the United States came from Europe, particularly from Italy, Poland, and Russia. There were also significant numbers of immigrants from Canada, Mexico, and other parts of Latin America. In addition, there were smaller numbers of immigrants from Asia and other parts of the world.

                Question: What does inioluwa mean,
                Answer: Inioluwa is a Nigerian name that means \"God's gift\" in Yoruba, a language spoken in Nigeria. It is typically given to a child who is seen as a blessing from God.

                Question: what is CVD,
                Answer: CVD stands for Cardiovascular Disease. It is a group of conditions that affect the heart and blood vessels. These conditions include coronary artery disease (which can lead to heart attacks),
                
                Now answer clearly, exactly the following: {text}.
                Answer: Let's think step by step before answering."""


    @overrides()
    def few_shots_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        prompt = self.few_shots_CoT(text)
        samples = self.functions.self_consistency(prompt, num_samples, max_len)
        if do_print:
            st.code(prompt, language='text')
        return samples
    @overrides()
    def few_shots_ToT(self, text):
      st.code('Simple QA do not need complicated ToT !',language="text")
    @overrides()
    def zero_shot_ToT(self, text):
      st.code('Simple QA do not need complicated ToT !',language="text")
    def run(self, text, do_print=False, type='Direct zero-shot', type_output = None, num_samples=5, max_len=200):
        prompt = None
        if type == 'Zero-shot CoT + Self-consistency':
            self.zero_shot_CoT_SC(text, num_samples, max_len, do_print)
            return 0
        elif type == 'Few-shots CoT + Self-consistency':
            self.few_shots_CoT_SC(text, num_samples, max_len, do_print)
        else:
            if type == 'Direct zero-shot':
                prompt = self.zero_shot_direct(text)
            elif type == 'Zero-shot CoT':
                prompt = self.zero_shot_CoT(text)
                max_len = 300
            elif type == 'Zero-shot CoT + Self-consistency':
                self.zero_shot_CoT_SC(text, num_samples, max_len, do_print)
                return 0
            elif type == 'Direct few-shots':
                prompt = self.few_shots_direct(text)
            elif type == 'Few-shots CoT':
                prompt = self.few_shots_CoT(text)
                max_len = 300
            if do_print:
                st.markdown("### Prompt:")
                st.code(prompt, language="text")
            input = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=False).to('cuda')
            input_len = input['input_ids'].shape[1]
            output = self.functions.generate_output(type=None, input=input, max_len=max_len)
            generate_out = output[0][input_len:]
            return self.tokenizer.decode(generate_out, skip_special_tokens=True)
