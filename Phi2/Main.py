from Phi2.basic_prompt_designs.Classfication import Classification
from Phi2.basic_prompt_designs.Computation import Computation
from Phi2.basic_prompt_designs.QA_knowledge import QA_knowledge
from Phi2.basic_prompt_designs.Reasoning import Reasoning
class Main:
    def __init__(self, tokenizer, model, name_task, type_prompt=None, comp_model=None, X=None, vectorizer=None, task_lib=None):
        self.name_task = name_task
        self.type_prompt = type_prompt
        if name_task=='classification':
            self.task = Classification(tokenizer,model)
        elif name_task=='computation':
            self.task = Computation(tokenizer, model, comp_model, X, vectorizer, task_lib)
        elif name_task=='reasoning':
            self.task = Reasoning(tokenizer, model, comp_model, X, vectorizer, task_lib)
        else:
            self.task = QA_knowledge(tokenizer, model, comp_model, X, vectorizer, task_lib)
    def main(self, text, name_prompt, do_print=False):
        return self.task.run(text, do_print, name_prompt)