from transformers import BertTokenizer, BertModel,BertForMaskedLM
from answer_evaluator import AnswerEvaluator
import torch
import numpy as np
from bert_score import BERTScorer

class Bert(AnswerEvaluator):

    def __init__(self):
        self.scoreList: list = None
    
    def set_setAnswer(self, setAnswer: str):
        return super().set_setAnswer(setAnswer)
    
    def set_genAnswer(self, genAnswer: str):
        return super().set_genAnswer(genAnswer)

    # generates 3-element list of mean scores
    def score(self) -> None:
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score([self.setAnswer], [self.genAnswer])
        self.scoreList = [P.mean(), R.mean(), F1.mean()]

    #gets prevision value for a given score    
    def getPrecision(self) -> float:
        if self.scoreList:
            return self.scoreList[0].numpy().item()
    
    #gets recall value for a given score
    def getRecall(self) -> float:
        if self.scoreList:
            return self.scoreList[1].numpy().item()

    #gets fmeasure value for a given score 
    def getfMeasure(self) -> float:
        if self.scoreList:
            return self.scoreList[2].numpy().item()

    def getAll(self) -> list:
        if self.scoreList:
            return [self.scoreList[0].numpy(), self.scoreList[1].numpy(), self.scoreList[2].numpy()]
        
    def getName(self) -> str:
        return "bertScore"