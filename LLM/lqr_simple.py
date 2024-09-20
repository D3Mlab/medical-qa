from llm_qa_reasoner import LLMQAReasoner

class SimpleReasoner(LLMQAReasoner):
  def __init__(self, llm, promptStyle):
    super().__init__(llm, promptStyle)

  def generateAnswer(self, docs, query):
     p = self.promptStyle.createPrompt(docs, query)
     a = self.llm.generate(p)
     self.output = a
     return a