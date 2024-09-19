from answer_evaluator import AnswerEvaluator
from rouge_score import rouge_scorer
from rouge_score_module import Rouge

class Rouge2(Rouge):
    def __init__(self):
        super().__init__()

    def set_setAnswer(self, setAnswer: str):
        return super().set_setAnswer(setAnswer)
    
    def set_genAnswer(self, genAnswer: str):
        return super().set_genAnswer(genAnswer)
        
    def score(self) -> None:
        return super().score() # dictionary will be in self.scoreDict
    
    def getPrecision(self) -> float:
        return super().getPrecision(self.getRouge2())
    
    def getRecall(self) -> float:
        return super().getRecall(self.getRouge2())
    
    def getfMeasure(self) -> float:
        return super().getfMeasure(self.getRouge2())
    
    def getAll(self) -> list:
        return super().getAll(self.getRouge2())
    
    def getName(self) -> str:
        return "rouge2"