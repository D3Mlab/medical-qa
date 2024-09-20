from prompt_type import PromptType

class QAPrompt(PromptType):

    def __init__(self, file_path, query, context):
       super().__init__()
       self.file_path = file_path
       self.data = {'query': query, 'context': context}
    