from ..snippet_embedder.vector_database import VectorDatabase
from InitialProject.QueryGenerate.qg_refined import ReFinEDRecognition
from InitialProject.QueryGenerate.qg_spacy import SpaCy
from InitialProject.QueryGenerate.qg_postag import POStag
from InitialProject.QueryGenerate.query_generator import QueryGenerator
from InitialProject.QueryGenerate.query_expander_llm import QueryExpander
from InitialProject.IR.ir_wiki_api import WikipediaAPIWrapper
from InitialProject.IR.bm25_scoring import BM25
from InitialProject.LLM.llm_wrapper import LLMWrapper
from InitialProject.LLM.lw_gpt import GPT
from InitialProject.LLM.ps_zeroshot import ZeroShot
from InitialProject.LLM.prompt_style import PromptStyle
from InitialProject.LLM.llm_qa_reasoner import LLMQAReasoner
from InitialProject.LLM.lqr_simple import SimpleReasoner
from transformers import pipeline



class SearchEngine:
    _database: VectorDatabase
    _query_generatorRe: QueryGenerator
    #To be cleaned up, they should be one query generator to rule them all
    _query_generatorSp: QueryGenerator
    _query_generatorNL: QueryGenerator
    _query_expander: QueryExpander
    _scoring: WikipediaAPIWrapper
    _llm: LLMWrapper
    _prompt_style: PromptStyle
    _reasoner: LLMQAReasoner
    _classifier: pipeline
    _threshold: float

    def __init__(self, api_key: str):
        self._database = VectorDatabase("ST")
        self._query_generatorRe = ReFinEDRecognition()
        self._query_generatorSp = SpaCy()
        self._query_generatorNL = POStag()
        self._query_expander = QueryExpander(4, api_key)
        self._scoring = BM25()
        self._llm = GPT(api_key)
        self._prompt_style = ZeroShot()
        self._reasoner = SimpleReasoner()
        self._classifier = pipeline("zero-shot-classification",
                                    model="facebook/bart-large-mnli")
        self._threshold = 0.8

    def initialize_database(self, docs: list[str], separate_type: str, saved_database_path: str = None):
        self._database.initializeVectorDataBase(docs, separate_type)

    def default_search(self, query: str, k: int) -> list[str]:
        return self._database.search_top_k(k, query)

    def hop_search(self, query: str, k: int, seg_style: str) -> list[str]:
        snippetList = []
        iterative_query = query
        self._scoring.set_k(3)
        #By default do 3 iterations
        for i in range(0, 3):
            self._query_generatorRe.setUserQuestion(iterative_query)
            self._query_generatorSp.setUserQuestion(iterative_query)
            self._query_generatorNL.setUserQuestion(iterative_query)
            self._query_expander.setUserQuestion(iterative_query)
            self._query_expander.queryList = list(set(self._query_generatorRe.generateIRQuery() +
                                                      self._query_generatorSp.generateIRQuery() +
                                                      self._query_generatorNL.generateIRQuery()))

            wiki_query = self._filter_keywords_for_relevancy(self._query_expander.generateIRQuery(), iterative_query)

            if wiki_query:
                docs = [t[1] for t in self._scoring.returnTopK(query, wiki_query)]
                if docs:
                    self._database.initializeVectorDataBase(docs, seg_style)
                    iterative_query = self._database.search_top_k(1, iterative_query)
                    snippetList.append(iterative_query[0])

        return snippetList

    def _filter_keywords_for_relevancy(self, potential_term: list[str], original_query: str) -> list [str] :
        scoring = self._classifier(original_query, potential_term)
        return [label for label, score in zip(scoring['labels'], scoring['scores']) if score > self._threshold]
