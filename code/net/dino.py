import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import Dinov2Model, AutoImageProcessor

class Dinov2Small(nn.Module):
    def __init__(self, embedding_size, pretrained=True, is_norm=True, bn_freeze=True):
        super(Dinov2Small, self).__init__()

        self.is_norm = is_norm
        self.embedding_size = embedding_size

        # Load pretrained DINOv2 backbone
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-small") if pretrained else Dinov2Model(Dinov2Model.config_class())

        # Feature size from CLS token
        self.num_ftrs = self.model.config.hidden_size

        # Projection head
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        output = torch.div(input, norm.view(-1, 1).expand_as(input))
        return output.view(input_size)

    def forward(self, x):
        # Get CLS token
        x = self.model(pixel_values=x).last_hidden_state[:, 0]  # shape: (B, hidden_size)

        x = self.model.embedding(x)

        if self.is_norm:
            x = self.l2_norm(x)

        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)
