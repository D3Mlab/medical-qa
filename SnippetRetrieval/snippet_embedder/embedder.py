from abc import ABCMeta, abstractmethod
from typing import Dict, Callable, List
import re

class Embedder(mataclass=ABCMeta):
    _snippetDispatchTable: Dict[str, Callable[[str], List[str]]]


    @abstractmethod
    def __init__(self):
        self._snippetDispatchTable = {
            'word': self._split_by_word,
            'sentence': self._split_by_sentence,
            'paragraph': self._split_by_paragraph
        }

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



