from torch import nn


class Trainer:
    def __init__(self, model: ):
        self.model = model()

    def __call__(self, *args, **kwargs):
        pass



class BaseModel(nn.Module):


    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def train(self):
        Trainer(self)()


class Model(BaseModel):
    def __init__(self):
        super().__init__()


    def __call__(self, x):
        x =


