from ir_resource import IRResource
import os
import json

class MedicalDataWrapper(IRResource):
  def __init__(self):
    self.returnedDocs = None
    super().__init__()

  def fetchDocs(self, folder_path):
    # Unicode escape sequences to remove from text
    esc_sequences = ['\xa0', '\xad', '\ue052', '\u202f', '\u200b']
    allDocs = []

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
      filepath = os.path.join(folder_path, filename)
      info = ""
      with open(filepath, "r", encoding="utf-8") as file_object:
        if "json" in folder_path:
          try:
            content = json.load(file_object)
            for key, value in content.items():
              info += value
              info += " "
          except json.JSONDecodeError: 
              continue
          filename = filename[:-5]
        else: #txt file
          info = file_object.read()
          filename = filename[:-4]
        
        for esc in esc_sequences:
            info = info.replace(esc, " ")


      allDocs.append((filename, info))

    self.returnedDocs = allDocs
    return allDocs