# Medical QA System Repository
This repository contains the code and dataset for the **Medical Question Answering (QA) system**. The system leverages a Retrieval-Augmented Generation (RAG) framework paired with LLM-based classifiers to generate answers to medical queries, tailoring responses based on the level of risk associated with each query. It is designed to be relevant for both research and real-world applications, such as healthcare support.
## Overview
The Medical QA system uses:
- **Query dataset**: Includes classification labels, raw documents, and gold standard responses (ground truths), as well as the names of the documents in which the ground truths are found.
- **RAG framework**: Combines information and snippet retrieval, as well as language models to answer medical queries.
- **Evaluation metrics**: Used to measure system performance using Rouge 1 Recall, Bert Precision, and Intersection Over Union (IoU) Score.

The dataset was curated from [The Heart Hub](https://ourhearthub.ca/) website, ensuring relevant and accurate information.
## Data Format
The dataset is provided in the following formats:
1. Dataset.csv: Contains the following columns:
    - **Question**: User input query.
    - **Query Type**: Indicates query classification label.
    - **Ground Truth**: Gold standard resonse extracted from The Heart Hub website.
    - **Document Source**: The document where the ground truth can be found.
2. Raw Documents:
    - Available in both ``.txt`` and ``.json`` formats.
    - The ``.json`` files include section-level segmentation for paragraph-level snippet testing.
## Setup and Usage
### Required Libraries
Install the following libraries to use the system:
```
pip install whoosh
pip install --upgrade openai
pip install rank_bm25
pip install faiss-gpu
pip install sentence_transformers
```
### System Configuration
The system’s main class can be defined as follows:
```
class MedQASystem(QASystem):
  def __init__(self, apiKey):
    # Options for IR and snippet retrieval: MedCPT | BioBERT | BioMedBERT | paraphrase-MiniLM-L6-v2 | gpt3 | TAS-B
    # Snippet retrieval also has a sparse retrieval option
    # The IR and snippet retrieval stages use the MedCPT cross encoder and the article encoder respectively
    self.irSys = BM25Local(folder_path='/content/medical-qa/OurHeartHub_txt', k=3) 
    self.vector_database = VectorDatabase("ST") 
    self.llm = GPT(apiKey)
    self.promptStyle = ZeroShot() # Options: ZeroShot, FewShot
    self.llmReasoner = SimpleReasoner(self.llm, self.promptStyle)
    self.Snippets = None

  def answer(self, query):
    self.queryGen.setUserQuestion(query)
    med_query = self.queryGen.generateIRQuery()
    fullArticles = self.irSys.query(query)
    fullArticles = [t[0] for t in fullArticles]

    fullArticlesList = []
    for articleTitle in fullArticles:
      file_path = f'/content/medical-qa/TheHeartHub_txt/{articleTitle}' 
      with open(file_path, 'r', encoding='utf-8') as file:
          articleContent = file.read()
      fullArticlesList.append(articleContent)

    if fullArticles:
      self.vector_database.initializeVectorDataBase(fullArticlesList, "paragraph")
      self.Snippets = self.vector_database.search_for_top_k(self.queryGen.userQuestion, 15)
      self.llmReasoner.generateAnswer(self.Snippets, self.queryGen.userQuestion)
      genAnswer = self.llmReasoner.output
      return genAnswer
```
### Adjusting Configurations
To modify the system to include only the Information Retrieval (IR) and answer generation stages, remove the lines initializing the vector database and snippet search, and directly pass the article content into the generateAnswer function.
```
if fullArticles:
    # Pass article content directly to the generateAnswer function
    self.llmReasoner.generateAnswer(fullArticlesList, self.queryGen.userQuestion)
    genAnswer = self.llmReasoner.output
    return genAnswer
```
## Evaluation
### Required Libraries
Install the following libraries to evaluate system performance:
```
pip install bert-score
pip install rouge-score
```
### Metrics
The evaluation metrics include:
- **Rouge 1 Recall**
- **Bert Precision**
- **IoU Score**
### Example Evaluation Code
```
for ind, row in new_result.iterrows():
  question = row['Question']
  groundtruth = row['GroundTruth']
  answer = row['Final Answer']
  
  common_substring = merge_common_substrings(groundtruth, answer)
  iou_score = iou_score_based_on_merged_substring(groundtruth, answer, common_substring)
  
  rouge1 = Rouge1()
  rouge1.set_genAnswer(answer)
  rouge1.set_setAnswer(groundtruth)

  bert = Bert()
  bert.set_genAnswer(answer)
  bert.set_setAnswer(groundtruth)

  scores = [
    rouge1.getPrecision(), rouge1.getRecall(), rouge1.getfMeasure(),
    bert.getPrecision(), bert.getRecall(), bert.getfMeasure()
  ]

  data.append([question, groundtruth, answer, iou_score] + scores)

columns = ["Questions", "GroundTruth", "Answer", "IoU Score", "Rouge1 Precision", "Rouge1 Recall", "Rouge1 fMeasure", "Bert Precision", "Bert Recall", "Bert fMeasure"]
df_result = pd.DataFrame(data, columns=columns)
```
