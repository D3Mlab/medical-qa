from answer_evaluator import AnswerEvaluator
from rouge_score import rouge_scorer
from rouge_score_module import Rouge

class RougeL(Rouge):
    def __init__(self):
        super().__init__()

    def set_genAnswer(self, genAnswer: str):
        return super().set_genAnswer(genAnswer)
    
    def set_setAnswer(self, setAnswer: str):
        return super().set_setAnswer(setAnswer)
        
    def score(self) -> None:
        return super().score() # dictionary will be in self.scoreDict
    
    def getPrecision(self) -> float:
        return super().getPrecision(self.getRougeL())
    
    def getRecall(self) -> float:
        return super().getRecall(self.getRougeL())
    
    def getfMeasure(self) -> float:
        return super().getfMeasure(self.getRougeL())
    
    def getAll(self) -> list:
        return super().getAll(self.getRougeL())
    
    def getName(self) -> str:
        return "rougeL"