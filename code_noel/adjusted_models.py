import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLayer(nn.Module):
    def __init__(self): 
        super().__init__()

        
    def forward(self, input_features):
        #input_features = input_features*0.5 # some calculations
        return input_features  

class NetworkWithCustomeLayer(nn.Module):
    def __init__(self, trained_model, layer_number):
        super(NetworkWithCustomeLayer, self).__init__()
        self.initial_layers =  nn.Sequential(*list(trained_model.children())[:layer_number])
        self.custom_layer = CustomLayer()
        self.last_layers =  nn.Sequential(*list(trained_model.children())[layer_number:-1])
        #self.fc = list(vgg_model.children())[-1]


    def forward(self, input):
        #print("input shape",input.shape)
        x_1 = self.initial_layers(input)
        #print("x1 shape",x_1.shape)
        x_c = self.custom_layer(x_1)
        #print("xc shape",x_c.shape)
        x_2 = self.last_layers(x_c)
        #print("x2 shape",x_2.shape)
        #x_3 = torch.flatten(x_2, 1)
        #print("x3 shape",x_3.shape)
        #x_4 = self.fc(x_3) 
        return x_2

def change_model(pretrained_model):
    adjusted_model = NetworkWithCustomeLayer(pretrained_model, 50)
    print("blablabla")

    print(dir(pretrained_model))
    print(dir(adjusted_model))
    #assert(False)
    return adjusted_model