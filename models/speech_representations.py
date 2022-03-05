from torch import nn
from torchaudio import transforms

#Residual blocks combined with BLSTM

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += res
        out = self.relu(out)
        return out


class ResidualBLSTM(nn.Module):
    def __init__(self, block, layers):
        super(ResidualBLSTM, self).__init__()
        self.in_channels=128
        self.speclayer = transforms.MelSpectrogram()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(1, self.in_channels, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer2 = self.make_layer(block, 128, layers[0])
        self.lstm = nn.LSTM(73395, 128, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32768, 128)
        self.classify = nn.Linear(128, 4)


    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = x.unsqueeze(1)
        in_spec = self.speclayer(x)
        out = self.initial_layer(in_spec)
        out_residual = self.layer2(out)
        batch, time = out_residual.size()[:2]
        out = out_residual.reshape(batch, time, -1)
        lstm_out, hidden = self.lstm(out)
        flat_out = self.flatten(lstm_out)
        fc = self.fc(flat_out)
        classify = self.classify(fc)
        return classify