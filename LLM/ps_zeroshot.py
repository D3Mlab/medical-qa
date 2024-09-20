from PromptType.pt_query import QAPrompt
from prompt_style import PromptStyle

class ZeroShot(PromptStyle):
  def __init__(self):
    super().__init__()
  
  def createPrompt(self, docs, query):
    context = ""
    for d in docs:
        context += d

    prompt_obj = QAPrompt("/content/nlqa-internal/InitialProject/PromptFiles/zero_shot_medical.jinja", query, context)
    prompt = prompt_obj.render_prompt()
    
    self.prompt = prompt
    return prompt