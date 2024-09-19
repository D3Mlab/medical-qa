from answer_evaluator import AnswerEvaluator
from rouge_score import rouge_scorer
from rouge_score_module import Rouge

class Rouge1(Rouge):
    def __init__(self):
        super().__init__()
    
    def set_genAnswer(self, genAnswer: str):
        return super().set_genAnswer(genAnswer)
    
    def set_setAnswer(self, setAnswer: str):
        return super().set_setAnswer(setAnswer)
    
    def score(self) -> None:
        return super().score() # dictionary will be in self.scoreDict
    
    def getPrecision(self) -> float:
        return super().getPrecision(self.getRouge1())
    
    def getRecall(self) -> float:
        return super().getRecall(self.getRouge1())
    
    def getfMeasure(self) -> float:
        return super().getfMeasure(self.getRouge1())
    
    def getAll(self) -> list:
        return super().getAll(self.getRouge1())
    
    def getName(self) -> str:
        return "rouge1"