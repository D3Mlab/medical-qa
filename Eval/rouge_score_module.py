from answer_evaluator import AnswerEvaluator
from rouge_score import rouge_scorer

class Rouge(AnswerEvaluator):
    def __init__(self):
        self.scoreDict: dict = None

    def set_genAnswer(self, genAnswer: str):
        return super().set_genAnswer(genAnswer)
    
    def set_setAnswer(self, setAnswer: str):
        return super().set_setAnswer(setAnswer)

    #initialize scorer and generate scores for all three rouge scoring systems as a dictionary
    def score(self) -> None:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.scoreDict = scorer.score(self.setAnswer, self.genAnswer)
    
    #returns rouge1
    def getRouge1(self) -> object:
        if self.scoreDict:
            return self.scoreDict["rouge1"]

    #returns rouge2
    def getRouge2(self) -> object:
        if self.scoreDict:
            return self.scoreDict["rouge2"]
    
    #returns rougeL
    def getRougeL(self) -> object:
        if self.scoreDict:
            return self.scoreDict["rougeL"]
    
    #gets precision value for a given score (e.g. score for rouge1)
    def getPrecision(self, Score: object) -> float:
        return Score[0]
    
    #gets recall value for a given score (e.g. score for rouge1)
    def getRecall(self, Score: object) -> float:
        return Score[1]   

    #gets fmeasure value for a given score (e.g. score for rouge1)
    def getfMeasure(self, Score: object) -> float:
        return Score[2]
    
    #returns all above scores (precision, recall ...) as a list
    def getAll(self, Score: object) -> list:
        if self.scoreDict:
            return [self.getPrecision(Score), self.getRecall(Score), self.getfMeasure(Score)]
    
    #prints all scores
    def printAll(self):
        if self.scoreDict:
            for key in self.scoreDict:
                print(f'{key}:\n{self.scoreDict[key]}')