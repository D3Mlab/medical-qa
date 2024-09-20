from typing import Dict, Callable, List
import re
import numpy as np
from torch.utils.data import TensorDataset
from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.utils.data
from tqdm import tqdm

class RoBERTa_Embedder:
    _snippetDispatchTable: Dict[str, Callable[[str], List[str]]]
    _tokenizer: RobertaTokenizer
    _roberta_model: RobertaModel

    def __init__(self):
        self._snippetDispatchTable = {
            'word': self._split_by_word,
            'sentence': self._split_by_sentence,
            'paragraph': self._split_by_paragraph
        }
        self._tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self._roberta_model = RobertaModel.from_pretrained('roberta-base')
        self._roberta_model.to('cuda' if torch.cuda.is_available() else 'cpu')

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
        # Check if CUDA is available and set the device accordingly
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Move the model to the device
        self._roberta_model = self._roberta_model.to(device)

        tokenized_texts = self._tokenizer.batch_encode_plus(
            texts,
            max_length=512,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors='pt',  # return PyTorch tensors
        )

        # Move the tensors to the device
        input_ids = tokenized_texts['input_ids'].to(device)
        attention_mask = tokenized_texts['attention_mask'].to(device)

        # Create a PyTorch DataLoader
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs)

        embeddings = []
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(dataloader, desc="Creating embeddings"):
                outputs = self._roberta_model(input_ids, attention_mask)
                last_hidden_states = outputs[0]  # Access the last hidden state
                doc_embedding = last_hidden_states.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(doc_embedding)
        return np.vstack(embeddings)

    def create_embedding(self, text: str) -> torch.Tensor:
        """
        Create an embedding for a single text
        :param text:
        :return:
        """
        embedding = self.create_batch_embeddings([text])
        embedding = torch.tensor(embedding).squeeze(0)
        return embedding

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
