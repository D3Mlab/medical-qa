import wikipediaapi
from ir_wiki_api import WikipediaAPIWrapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy

class CosineSimilarityScorer(WikipediaAPIWrapper):
  def __init__(self):
    self.vectorizer = TfidfVectorizer()
    super().__init__()

  def tf_idf_fit(self, stringList: list) -> object:
    tf_idf_matrix = self.vectorizer.fit_transform(stringList)
    return tf_idf_matrix
  
  def tf_idf(self, stringList: list) -> object:
    tf_idf_matrix = self.vectorizer.transform(stringList)
    return tf_idf_matrix
  
  def cosineSimilarity(self, tfidf_query: object, tfidf_docs: object) -> list:
    return cosine_similarity(tfidf_query, tfidf_docs).flatten()

  def returnTopK(self, prompt: str, queryTerms: list) -> list:
    allDocs = self.fetchDocs(queryTerms)
    tfidf_docs = self.tf_idf_fit(doc[1] for doc in allDocs)
    tfidf_query = self.tf_idf([prompt])
    similarity_score = self.cosineSimilarity(tfidf_query, tfidf_docs)
    similarity_score = similarity_score.argsort()[::-1]
    k = min(self.k, len(allDocs))
    topkDocs = [allDocs[i] for i in similarity_score[:k]]
    return topkDocs
    