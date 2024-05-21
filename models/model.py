import torch.nn as nn
import torch
from torch.autograd import Function


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.nn_model = nn.Sequential(
            nn.Linear(input_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, output_dim)
        )

    def forward(self, text):
        output = self.nn_model(text)
        return output



class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class DANN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DANN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256, bias=True),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            #nn.LeakyReLU(inplace=True),
            #nn.Linear(128, 128)
            #nn.Dropout(0.5),
            #nn.LeakyReLU(inplace=True)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            nn.BatchNorm1d(128),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, output_dim)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            #nn.BatchNorm1d(128),
            #torch.nn.Sigmoid()
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2),
            #torch.nn.Sigmoid()
            nn.LogSoftmax(dim=0)
        )

    
    def grad_reverse(self, x):
        return GradReverse.apply(x)

    def forward(self, input_data):
        feature = self.feature(input_data)
        reverse_feature = self.grad_reverse(feature)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature).squeeze()

        #print(domain_output.shape)

        return class_output, domain_output
