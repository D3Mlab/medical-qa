from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
from llm_wrapper import LLMWrapper

class Llama(LLMWrapper):
  def __init__(self):
    super().__init__()
  
  def load_model(self):
    '''
    Loads the model once the method is called to save GPU RAM to allow
    for many prompt calls
    '''
    self.model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length = 8192,
        load_in_4bit = True,
    )

    self.tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3",
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, 
    )
    FastLanguageModel.for_inference(self.model)
    
  def generate(self, prompt):
    # Passing the prompt into Llama3
    messages = [
        {"from": "human", "value": prompt},
    ]
    inputs = self.tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")

    text_streamer = TextStreamer(self.tokenizer)
    response = self.model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)

    # Isolate the answer from the output using string processing
    response_text = self.tokenizer.batch_decode(response)

    response_string = response_text[0]

    # Find the second instance of the delimeters marking the beginning and end of a section
    answer_start = "<|end_header_id|>"
    answer_end = "<|eot_id|>"

    start_index1 = response_string.find(answer_start)
    end_index1 = response_string.find(answer_end)

    start_index2 = response_string.find(answer_start, end_index1 + 1)
    end_index2 = response_string.find(answer_end, start_index2 + 1)

    # Define the substring representing the LLM output
    output = response_string[start_index2 + len(answer_start)+2:end_index2]
    self.output = output
    return output