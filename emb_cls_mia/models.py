import torch
import torch.nn as nn


class GDClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(GDClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(64,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh()
        )
        self.classifier = nn.Linear(32,num_classes)
        
    def forward(self,x):
        hidden_out = self.features(x)
        out = self.classifier(hidden_out)
        return out