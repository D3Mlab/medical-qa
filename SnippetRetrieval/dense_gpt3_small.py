from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
import numpy as np
from abc import ABCMeta, abstractmethod
from nltk.tokenize import sent_tokenize
from snippet_retriever import SnippetRetriever
from openai import OpenAI


class DenseRetrievergpt3(SnippetRetriever):
    def __init__(self, snippetType, k, apiKey):
        super().__init__(snippetType, k)
        self.model = 'text-embedding-3-small' ###
        self.client = OpenAI(api_key = apiKey)
        self.documents = []
        self.snippets = []
        self.embeddings = []

    def get_embeddings(self, texts, model="text-embedding-3-small"):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=text,
                model=model
            ).data[0].embedding
            embeddings.append(response)
        return np.array(embeddings)


    def split(self, documents): 
        split_methods = {
            "Section": lambda x: x.split("\n\n"),
            "Sentence": sent_tokenize,
            "Paragraph": lambda x: x.split("\n")
        }
        allSnippets = []
        for doc in documents:
            snippets = split_methods[self.snippetType](doc)
            allSnippets.extend(snippets)
        self.snippets = allSnippets

    def returnTopK(self, docs, query):
        self.documents = [doc[1] for doc in docs]  # Assuming docs are tuples of (title, content)
        self.split(self.documents)
        query_emb = self.get_embeddings([query]) # Assuming single query
        doc_emb = self.get_embeddings(self.snippets)
        print("query_emb:", query_emb[0].size)
        print("doc_emb:", doc_emb[0].size)

        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

        doc_score_pairs = list(zip(self.snippets, scores))
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        res = []
        count = 0
        for doc, score in sorted_pairs:
            print(doc)
            res.append(doc)
            count += 1
            if count == self.k:
                break

        print(f"Number of snippets: {len(self.snippets)}, Number of embeddings: {len(doc_emb)}")

        return res
