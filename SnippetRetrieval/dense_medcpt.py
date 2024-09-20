import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from snippet_retriever import SnippetRetriever
from nltk.tokenize import sent_tokenize

class DenseRetrieverMedCPT(SnippetRetriever):
    def __init__(self, snippetType, k, doc_emb):
        super().__init__(snippetType, k)
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.documents = []  # this should be a list of document contents
        self.snippets = []  # this will store snippets of documents
        self.doc_emb = doc_emb  # Tensor of precomputed document embeddings

    def split(self):
        allSnippets = []
        split_methods = {"Section": lambda x: x.split("\n\n"), "Sentence": sent_tokenize, "Paragraph": lambda x: x.split("\n")}
        for d in self.documents:
            snippets = split_methods[self.snippetType](d)
            allSnippets.extend(snippets)
        self.snippets = allSnippets

    def encode(self, texts):
        # Tokenize and encode the texts
        with torch.no_grad():
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            output = self.model(**encoded_input)
            embeddings = output.last_hidden_state[:, 0, :]
            return embeddings

    def returnTopK(self, docs, query):

        self.documents = docs
        self.split()
        print(self.snippets)
        # Encode the query to get the query embedding
        query_emb = self.encode([query])[0]

        # Compute cosine similarities between query embedding and document embeddings
        scores = F.cosine_similarity(query_emb.unsqueeze(0), self.doc_emb).cpu().tolist()

        # Combine snippets & scores
        doc_score_pairs = list(zip(self.snippets, scores))

        # Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # Output the top K passages & scores
        res = []
        count = 0
        for doc, score in doc_score_pairs:
            res.append((doc, score))
            count += 1
            if count == self.k:
                break

        return res