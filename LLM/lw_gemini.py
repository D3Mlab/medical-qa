import google.generativeai as genai
from llm_wrapper import LLMWrapper

class Gemini(LLMWrapper):
  def __init__(self, apikey):
    super().__init__()
    # Pass in Google API key
    genai.configure(api_key=apikey)
    
  def generate(self, prompt):
    # Prompts Gemini and returns the result
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    output = response.text
    self.output = output
    return output