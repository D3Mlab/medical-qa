from abc import ABCMeta, abstractmethod

class QASystem(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def answer(self):
        pass