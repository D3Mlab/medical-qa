import torch
from ir_medical import MedicalDataWrapper
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class MedCPT_Medical(MedicalDataWrapper):
    def __init__(self, folder_path, k):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        self.model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        self.folder_path = folder_path
        self.k = k
    
    def fetchDocs(self) -> list:
        super().fetchDocs(self.folder_path)
        self.docs = [doc[1] for doc in self.returnedDocs]
        self.file_names = [doc[0] for doc in self.returnedDocs]

    def returnTopK(self, prompt: str) -> list:
        topKDocs = []
    
        if self.returnedDocs:
            pairs = [[prompt, article] for article in self.docs]
            with torch.no_grad():
                encoded = self.tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
            logits = self.model(**encoded).logits.squeeze(dim=1)
            sorted_data_desc, indices_desc = torch.sort(logits, descending=True)

            k = min(self.k, len(self.returnedDocs))
            for i in range(k): 
                topKDocs.append(self.file_names[indices_desc[i]])
       
        return topKDocs