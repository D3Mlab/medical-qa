from whoosh import index, writing
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import *
from whoosh.qparser import QueryParser
from whoosh import qparser
from pathlib import Path
import tempfile
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from snippet_retriever import SnippetRetriever

class SparseRetrieval(SnippetRetriever):
  def __init__(self, snippetType, k):
    super().__init__(snippetType, k)
    self.snippets = None
    self.scoredSnippets = None
    self.relevantSnippets = None

  def split(self, documents):
    allSnippets = []

    split_methods = {"Section": lambda x: x.split("\n\n"), "Sentence": sent_tokenize, "Paragraph": lambda x: x.split("\n")}

    for d in documents:
        pageTitle = d[0]
        snippets = split_methods[self.snippetType](d[1])
        for idx, val in enumerate(snippets):
            allSnippets.append((f"{pageTitle}_{idx}", val))
         
    self.snippets = allSnippets
  
  #all following functions are adapted from 451 lab 1
  def create_index(self):
      """
      INPUT:
          None
      OUTPUT:
          None

      NOTE: Please update self.index_sys which should have type whoosh.index.FileIndex
      """
      # Generate a temporary directory for the index
      indexDir = tempfile.mkdtemp()

      # first, define a Schema for the index
      mySchema = Schema(doc_id = ID(stored=True),
                doc_content = TEXT(analyzer = RegexTokenizer()))

      self.index_sys = index.create_in(indexDir, mySchema)

  def addFilesToIndex(self):
      # open writer
      writer = writing.BufferedWriter(self.index_sys, period=None, limit=1000)

      try:
          # write each file to index
          for s in self.snippets:
              writer.add_document(doc_id = s[0], 
                                  doc_content = s[1])
      finally:
          # close the index
          writer.close()

  def create_parser_searcher(self):
      """
      INPUT:
          None
      OUTPUT:
          None

      NOTE: Please update self.query_parser and self.self.searcherwhich should have type whoosh.qparser.default.QueryParser and whoosh.searching.Searcher respectively
      """
      self.query_parser = QueryParser("doc_content", schema=self.index_sys.schema, group=qparser.OrGroup)
      self.searcher = self.index_sys.searcher()

  def perform_search(self, search_query):
      """
      INPUT:
          search_query: string
      OUTPUT:
          topicResults: whoosh.searching.Results

      NOTE: Utilize self.query_parser and self.searcher to calculate the result for search_query
      """

      query = self.query_parser.parse(search_query)
      results = self.searcher.search(query, limit=None)
      self.scoredSnippets = results
      return results

  def returnTopK(self, docs, query):

      self.split(docs)
      self.create_index()
      self.addFilesToIndex()
      self.create_parser_searcher()
      self.perform_search(query)

      localK = int(self.k)
      relevant_docs = []
      for doc, score in self.scoredSnippets.items():
          relevant_docs.append(self.snippets[doc][1])
          localK -= 1
          if localK == 0:
            break
      self.relevantSnippets = relevant_docs
      return relevant_docs
  
  def check_index(self, words=None):
      print(f"Docs indexed: {self.index_sys.doc_count()}")

      if words:
          reader = self.index_sys.reader()
          for word in words:
              print(f"# docs with '{word}'", reader.doc_frequency("doc_content", word))