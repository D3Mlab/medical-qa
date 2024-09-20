from ir_medical import MedicalDataWrapper
from rank_bm25 import BM25Okapi
import numpy as np

class BM25_Medical(MedicalDataWrapper):
    def __init__(self, folder_path, k):
        super().__init__()
        self.bm25 = None
        self.folder_path = folder_path
        self.k = k
    
    def fetchDocs(self) -> list:
        super().fetchDocs(self.folder_path)

        self.docs = [doc[1].lower().split() for doc in self.returnedDocs]
        self.bm25 = BM25Okapi(self.docs)

    def returnTopK(self, prompt: str) -> list:
        topKDocs = None
        if self.returnedDocs:
            promptToks = prompt.lower().split()
            scores = self.bm25.get_scores(promptToks)
            k = min(self.k, len(self.returnedDocs))
            topKscores = np.argsort(scores)[::-1][:k]
            topKDocs = [self.returnedDocs[i] for i in topKscores]
       
        return topKDocs