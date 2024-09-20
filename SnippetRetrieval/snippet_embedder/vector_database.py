import numpy as np
import faiss
import torch
import os

from bert_embedder import BERTEmbedder
from XLNet_embedder import XLNet_Embedder
from RoBERTa_embedder import RoBERTa_Embedder
from ST_embedder import ST_Embedder
from vector_database_creator import VectorDatabaseCreator


class VectorDatabase:
    _snippets: list[str] = []
    _index: faiss.Index
    _embedder: BERTEmbedder | XLNet_Embedder | RoBERTa_Embedder | ST_Embedder
    _vector_database_creator: VectorDatabaseCreator

    def __init__(self, embedder: str):
        """
        Initialize the VectorDatabase with the specified embedder
        :param embedder:
        """
        if embedder == 'BERT':
            self._embedder = BERTEmbedder()
        elif embedder == 'XLNet':
            self._embedder = XLNet_Embedder()
        elif embedder == 'RoBERTa':
            self._embedder = RoBERTa_Embedder()
        elif embedder == 'ST':
            self._embedder = ST_Embedder()
        self._vector_database_creator = VectorDatabaseCreator()

    def initializeVectorDataBase(self, docs: list[str], type_str: str):
        """
        Initialize the VectorDatabase with the specified documents and type
        :param docs:
        :param type_str:
        :return:
        """
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self._snippets = self._embedder.split_snippet(docs, type_str)
        snippet_embedding: list[torch.tensor] = self._embedder.create_batch_embeddings(self._snippets)
        self._index = self._vector_database_creator.create_database(snippet_embedding)

    def search_for_top_k(self, query: str, k: int) -> list[str]:
        """
        Search for the top k relevant snippets, input is the query(string)
        :param query:
        :param k:
        :return:
        """
        query_embedding = self._embedder.create_embedding(query).reshape(-1, self._index.d)
        # if the query is not a np array, convert it to a np array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding_np = query_embedding.cpu().detach().numpy()
        else:
            query_embedding_np = query_embedding
        top_k_snippet = self.get_top_K_relavent_snippet(k, query_embedding_np)
        return top_k_snippet

    def get_top_K_relavent_snippet(self, k: int, query_embedding: np.ndarray) -> list[str]:
        """
        Get the top k relevant snippets, input is the query embedding
        :param k:
        :param query_embedding:
        :return:
        """
        top_k_snippet: list[str] = []
        faiss.normalize_L2(query_embedding)
        distance, indices = self._index.search(query_embedding, k)
        indices = indices[0]
        #get the actual index of the top k snippet embeddings
        for i in range(k):
            top_k_snippet.append(self._snippets[indices[i]])
        return top_k_snippet

    def search_top_k(self, k: int, query: str) -> list[str]:
        query_embedding = self._embedder.create_embedding(query).reshape(-1, self._index.d)
        # if the query is not a np array, convert it to a np array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.cpu().detach().numpy()
        top_k_snippet: list[str] = []
        faiss.normalize_L2(query_embedding)
        distance, indices = self._index.search(query_embedding, k)
        indices = indices[0]
        #get the actual index of the top k snippet embeddings
        for i in range(k):
            top_k_snippet.append(self._snippets[indices[i]])
        return top_k_snippet
