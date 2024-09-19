from abc import ABCMeta, abstractmethod

class AnswerEvaluator(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set_genAnswer(self, genAnswer: str):
        self.genAnswer = genAnswer
    
    @abstractmethod
    def set_setAnswer(self, setAnswer: str):
        self.setAnswer = setAnswer

    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def getPrecision(self) -> float:
        pass

    @abstractmethod
    def getRecall(self) -> float:
        pass

    @abstractmethod
    def getfMeasure(self) -> float:
        pass

    @abstractmethod
    def getAll(self) -> list:
        pass

    @abstractmethod
    def getName(self) -> str:
        pass