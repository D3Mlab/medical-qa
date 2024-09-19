from abc import ABCMeta, abstractmethod
from Main.qa_system import QASystem

class QAEvaluator(metaclass=ABCMeta): 

    @abstractmethod
    def __init__(self, System: QASystem):
        self.System = System

    @property
    @abstractmethod
    def scoresDf(self):
        pass

    @scoresDf.setter
    @abstractmethod
    def scoresDf(self, data):
        pass

    @property
    @abstractmethod
    def metrics(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def calculate_metrics(self):
        pass

    