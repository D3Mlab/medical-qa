from abc import ABCMeta, abstractmethod

class IRResource(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self.k = None
        self.returnedDocs = None

    def set_k(self, value : int) -> None:
        self.k = value

    # @property
    # def k(self):
    #     return self.k
    
    # @k.setter
    # def k(self, value : int) -> None:
    #     self.k = value
    

    def fetchDocs(self):
        pass

    def scoreDocs(self):
        pass
