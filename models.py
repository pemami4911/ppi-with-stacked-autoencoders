import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedAutoencoder(nn.Module):
    """
    1-hidden layer AE trained with MSE loss
    """
    def __init__(self, input_size, hidden_layer_size):
        super(StackedAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_layer_size)
        self.decoder = nn.Linear(hidden_layer_size, input_size)

    def embedding(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid(x)
        return self.decoder(x)

class Classifier(nn.Module):
    """
    Fully-connected NN binary classifier, trained with BCEWithLogitsLoss
    """
    def __init__(self, sae, hidden_layer_size):
        super(Classifier, self).__init__()
        self.sae = sae
        self.out = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        x = self.sae.embedding(x)
        x = F.relu(x)
        x = self.out(x)
        return x.squeeze()
