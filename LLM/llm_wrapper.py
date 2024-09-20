from abc import ABCMeta, abstractmethod

class LLMWrapper(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
      pass

    @abstractmethod
    def generate(self):
      pass