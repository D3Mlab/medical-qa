from PromptType.pt_entities import EntityPrompt
from prompt_style import PromptStyle

class MedClassifier(PromptStyle):
  def __init__(self):
    super().__init__()
  
  def createPrompt(self, query):

    prompt_obj = EntityPrompt("/content/nlqa-internal/InitialProject/PromptFiles/medical_classifier.jinja", query)
    prompt = prompt_obj.render_prompt()
    
    self.prompt = prompt
    return prompt