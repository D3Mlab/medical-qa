import torch

from bert_embedder import BERTEmbedder
import numpy as np
import faiss
from tqdm import tqdm


class VectorDatabaseCreator:
    _sample: str

    def __init__(self):
        pass

    def create_database(self, snippet_embeddings: list[torch.tensor]) -> faiss.Index:
        """
        Create a faiss index for the snippet embeddings provided
        :param snippet_embeddings:
        :return:
        """
        dimension_size = 768
        faiss.normalize_L2(snippet_embeddings)
        index = faiss.IndexFlatIP(dimension_size)
        index.add(snippet_embeddings)
        return index
