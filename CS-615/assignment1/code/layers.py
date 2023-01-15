from abc import ABC, abstractclassmethod

class Layer(ABC):
    def __init__(self) -> None:
        self.__prevIn = []
        self.__prevOut=[]

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut

    @abstractclassmethod
    def forward(self, dataIn):
        pass

    @abstractclassmethod
    def gradient(self):
        pass

    @abstractclassmethod
    def backwards(self, gradIn):
        pass

