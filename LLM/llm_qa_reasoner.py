from abc import ABCMeta, abstractmethod

class LLMQAReasoner(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, llm, promptStyle):
        self.llm = llm
        self.promptStyle = promptStyle

    @abstractmethod
    def generateAnswer(self):
        pass