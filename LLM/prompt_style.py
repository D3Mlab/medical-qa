from abc import ABCMeta, abstractmethod

class PromptStyle(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
      pass

    @abstractmethod
    def createPrompt(self):
      pass