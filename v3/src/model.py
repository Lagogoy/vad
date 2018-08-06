# pylint:disable=E1101, E1102

import torch
import torch.nn as nn
import math

class lstmNet(nn.Module):
    def __init__(self, input_dim, batch_size = 1):
        super(lstmNet, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim1 = 30
        self.hidden_dim2 = 20       
        self.linear_dim1 = 40
        self.linear_dim2 = 10
        # self.linear_dim3 = 1
        self.init_weights()

        self.lstm1 = nn.LSTM(input_dim, self.hidden_dim1, bidirectional = True)
        self.lstm2 = nn.LSTM(self.hidden_dim1*2, self.hidden_dim2, bidirectional = True)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim2*2, self.linear_dim1),
            nn.ReLU(), # nn.Tanh(),
            nn.Linear(self.linear_dim1, self.linear_dim2),
            nn.ReLU(), # nn.Tanh(),
            nn.Linear(self.linear_dim2, 1),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, sentence):
        hidden1 = (torch.zeros(2, self.batch_size, self.hidden_dim1).cuda(),
                   torch.zeros(2, self.batch_size, self.hidden_dim1).cuda())
        hidden2 = (torch.zeros(2, self.batch_size, self.hidden_dim2).cuda(),
                   torch.zeros(2, self.batch_size, self.hidden_dim2).cuda())
        # print("x: ", type(sentence), sentence.shape)
        out, hidden1 = self.lstm1(sentence, hidden1)
        out, hidden2 = self.lstm2(out, hidden2)
        out = self.classifier(out)
        return out

# Simple test
if __name__ == "__main__":
    x = torch.tensor(torch.randn(3,1,5)).float().cuda()
    batch_size = x.shape[1]
    input_dim = x.shape[2]

    model = lstmNet(input_dim, batch_size).cuda()
    print(model)

    out = model(x)
    print("out:", out)
    print(type(out), out.shape)
