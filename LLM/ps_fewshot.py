from PromptType.pt_query import QAPrompt
from prompt_style import PromptStyle

class FewShot(PromptStyle):
  def __init__(self):
    super().__init__()
  
  def createPrompt(self, docs, query, classification):
    context = ""
    for d in docs:
        context += d

    prompt_obj = QAPrompt("/content/nlqa-internal/InitialProject/PromptFiles/twoshot_medical.jinja", query, classification, context)
    prompt = prompt_obj.render_prompt()

    self.prompt = prompt
    return prompt