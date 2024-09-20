from qa_system import QASystem
from LLM.ps_zeroshot import ZeroShot
# from LLM.ps_fewshot import FewShot
from LLM.lw_gpt import GPT
# from LLM.lw_gemini import Gemini
# from LLM.lw_llama import Llama
from LLM.ps_medClassifier import MedClassifier
from LLM.lqr_simple import SimpleReasoner
from LLM.med_qa_reasoner import MedReasoner
from IR.bm25_medical import BM25_Medical
from IR.medcpt_medical import MedCPT_Medical
from IR.ir_medical import MedicalDataWrapper
from SnippetRetrieval.dense_gpt3_small import DenseRetrievergpt3
import os


class MedQASystem(QASystem):
  def __init__(self, apiKey):
    self.classification = MedClassifier()
    self.snippetRetrv = DenseRetrievergpt3("Sentence", 30, apiKey)
    self.docs = MedCPT_Medical("./TheHeartHub_txt", 5)
    self.llm = GPT(apiKey)
    self.promptStyle = ZeroShot()
    self.llmReasoner = SimpleReasoner(self.llm, self.promptStyle)
    self.llmClassifier = MedReasoner(self.llm, self.classification)
    self.Snippets = None

  def answer(self, query):
    # Classify the question
    classification = self.llmClassifier.generateAnswer(query)
    print(classification)

    if "yes" in classification:
      self.category = "answerable"
    elif "no" in classification:
      self.category = "unanswerable"
    elif "helpful deferral" in classification:
      self.category = "helpful deferral"
    else:
      self.category = "unknown"

    # Read text files from a specified folder
    folder_path = "./TheHeartHub_txt"
    articles = []
    file_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                articles.append(file.read())
                file_names.append(filename)

    docs = []
    docs = self.docs.returnTopK(query, articles, 5)

    fullArticlesList = []
    for articleTitle in docs:
      file_path = f'/content/nlqa-internal/InitialProject/OurHeartHub_txt/{articleTitle}'
      with open(file_path, 'r', encoding='utf-8') as file:
          articleContent = file.read()
      fullArticlesList.append([articleTitle, articleContent])

    if docs:
      # Generate snippets
      self.Snippets = self.snippetRetrv.returnTopK(fullArticlesList, query)
      # Generate answer
      genAnswer = self.llmReasoner.generateAnswer(self.Snippets, query, self.category)
      genAnswer = self.llmReasoner.output

      return genAnswer