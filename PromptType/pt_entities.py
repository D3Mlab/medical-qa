from prompt_type import PromptType

class EntityPrompt(PromptType):

    def __init__(self, file_path, user_question):
       super().__init__()
       self.file_path = file_path
       self.data = {'userQuestion': user_question}
