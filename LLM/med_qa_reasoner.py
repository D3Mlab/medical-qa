from llm_qa_reasoner import LLMQAReasoner

class MedReasoner(LLMQAReasoner):
  def __init__(self, llm, promptStyle):
    super().__init__(llm, promptStyle)

  def generateAnswer(self, query):
     p = self.promptStyle.createPrompt(query)
     a = self.llm.generate(p)
     self.output = a
     return a