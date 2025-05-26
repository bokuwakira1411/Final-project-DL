from abc import ABC, abstractmethod

class Pattern(ABC):
    @abstractmethod
    def zero_shot_direct(self, text):
        pass
    @abstractmethod
    def zero_shot_CoT(self, text):
        pass
    @abstractmethod
    def zero_shot_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        pass
    @abstractmethod
    def zero_shot_ToT(self, text):
        pass
    @abstractmethod
    def few_shots_direct(self, text):
        pass
    @abstractmethod
    def few_shots_CoT(self, text):
        pass
    @abstractmethod
    def few_shots_CoT_SC(self, text, num_samples=5, max_len=50, do_print=False):
        pass
    @abstractmethod
    def few_shots_ToT(self, text):
        pass