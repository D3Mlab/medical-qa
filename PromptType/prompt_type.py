from jinja2 import Template
from abc import ABCMeta, abstractmethod

class PromptType(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
       pass
    
    def render_prompt(self):
      with open(self.file_path, 'r') as file:
          prompt_template = file.read()

      template = Template(prompt_template)
      prompt = template.render(self.data)
      return prompt