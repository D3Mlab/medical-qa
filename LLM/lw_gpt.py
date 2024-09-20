from openai import OpenAI
from llm_wrapper import LLMWrapper

class GPT(LLMWrapper):
  def __init__(self, apiKey):
    self.client = OpenAI(api_key=apiKey)
    super().__init__()
    
  def generate(self, prompt, temp=0.2):
    response = self.client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role":"system", "content":"You are a helpful assistant."}, {"role":"user", "content": prompt}], 
            temperature=temp)
    output = response.choices[0].message.content
    self.output = output
    return output