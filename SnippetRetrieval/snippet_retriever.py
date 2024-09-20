from abc import ABCMeta, abstractmethod
import torch
import numpy as np

class SnippetRetriever(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, snippetType, k):
        self.k = k
        self.snippetType = snippetType
        self.docs = None

    def add_documents(self, docs):
        self.documents.extend(docs)
        embeddings = []
        with torch.no_grad():
            for doc in docs:
                inputs = self.tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512)
                outputs = self.model(**inputs)
                doc_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(doc_embedding)
        self.embeddings.append(np.vstack(embeddings))

    @abstractmethod
    def returnTopK(self):
        pass
