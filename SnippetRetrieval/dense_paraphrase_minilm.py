from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from snippet_retriever import SnippetRetriever


class DenseRetrieverMiniLM(SnippetRetriever):
    def __init__(self, snippetType, k):
        super().__init__(snippetType, k)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.documents = [] 
        self.snippets = [] 
        self.embeddings = []
        
    def split(self):
        allSnippets = []
        split_methods = {"Section": lambda x: x.split("\n\n"), "Sentence": sent_tokenize, "Paragraph": lambda x: x.split("\n")}

        for d in self.documents:
            snippets = split_methods[self.snippetType](d[1])
            allSnippets.extend(snippets)
        
        self.snippets = allSnippets


    def returnTopK(self, docs, query):
        
        # clean fields
        self.snippets = []
        self.embeddings = []

        # pass in docs
        self.documents = docs
        
        self.split()
        print(self.snippets)
        query_emb = self.model.encode(query)
        print("query_emb:", query_emb)
        doc_emb = self.model.encode(self.snippets)
        print("doc_emb:", doc_emb)

        #Compute dot score between query and all document embeddings
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

        #Combine docs & scores
        doc_score_pairs = list(zip(self.snippets, scores))

        #Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        #Output passages & scores
        res = []
        count = 0
        for doc, score in doc_score_pairs:
            print(doc)
            res.append(doc)
            count += 1
            if count == self.k:
                break

        print(f"Number of snippets: {len(self.snippets)}, Number of embeddings: {len(doc_emb)}")
        
        return res