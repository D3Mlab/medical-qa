from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm
from typing import Dict, Callable, List
import re
import torch
from tqdm import tqdm


class ST_Embedder:
    _snippetDispatchTable: Dict[str, Callable[[str], List[str]]]
    _model: SentenceTransformer

    def __init__(self):
        self._snippetDispatchTable = {
            'word': self._split_by_word,
            'sentence': self._split_by_sentence,
            'paragraph': self._split_by_paragraph
        }
        self._model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        # Enable GPU if available
        self._model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def split_snippet(self, docs: list[str], type_str: str):
        snippets: list[str] = []
        split_method = self._snippetDispatchTable.get(type_str)
        if split_method is None:
            raise ValueError(f"Unknown type '{type_str}'. Valid options are 'word', 'sentence', 'paragraph'.")
        for doc in docs:
            snippets.extend(split_method(doc))
        return snippets

    def _split_by_word(self, doc: str) -> list[str]:
        return doc.split()

    def _split_by_sentence(self, doc: str) -> list[str]:
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
        return re.split(sentence_pattern, doc)

    def _split_by_paragraph(self, doc: str) -> List[str]:
        paragraphs = doc.split('\n')
        return [para for para in paragraphs if para.strip() != '']

    def create_batch_embeddings(self, texts: list[str], bs=48) -> np.ndarray:
        """
        Create embeddings for a list of texts
        :param texts:
        :param bs:
        :return:
        """
        embeddings = self._model.encode(texts)
        return embeddings

    def create_embedding(self, text: str) -> torch.Tensor:
        """
        Create an embedding for a single text
        :param text:
        :return:
        """
        return self._model.encode(text)

    def convert_snippet_to_embedding(self, snippets: list[str]):
        """
        Convert a list of snippets to a list of embeddings
        :param snippets:
        :return:
        """
        embedding: list[torch.tensor] = []
        for i in tqdm(range(0, len(snippets))):
            embedding.extend(self.create_embedding(snippets[i]))
        return embedding
