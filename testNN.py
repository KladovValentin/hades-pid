from models.model import DANN
from models.model import Model
import torch
import torch.nn as nn

def loadModel(input_dim, nClasses,):
    nn_model = DANN(input_dim=input_dim, output_dim=nClasses).type(torch.FloatTensor)
    nn_model.type(torch.FloatTensor)
    nn_model.load_state_dict(torch.load('modelScr.pt'))
    return nn_model

def makePredictionList(dfTable, )