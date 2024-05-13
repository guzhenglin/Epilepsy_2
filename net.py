import torch
import torch.nn as nn
import torch.nn.functional as F

class net_freq60_1d_nopad(nn.Module):
    def __init__(self, out_channels1, out_channels2, out_channels3, out_channels4, lstm_hidden, lstm_step, fc1_out):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.fc1_mlp = nn.Linear(18,4)
        self.fc1_mlp = nn.Conv2d(18, 4, 1, bias=False)
        # self.fc2_mlp = nn.Linear(4, 18)
        self.fc2_mlp = nn.Conv2d(4, 18, 1, bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=18, out_channels=out_channels1, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(out_channels2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels2, out_channels=out_channels3, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(out_channels3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels3, out_channels=out_channels4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(out_channels4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.lstm = nn.LSTM(input_size=out_channels4, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(lstm_hidden*lstm_step*2, fc1_out)    # lstm_hidden*lstm_step
        self.fc2 = nn.Linear(fc1_out, 16)
        self.fc3 = nn.Linear(16, 8)
        # self.fc4 = nn.Linear(36, 18)
        self.dropout = nn.Dropout(0.3)
        self.fc5 = nn.Linear(8, 4)

    def forward(self, x):
        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1, x.shape[3])
        # print("1:", x.shape)
        x = x.permute(0, 2, 1, 3)
        # x = torch.squeeze(x)
        # print("2:", x[0], x.shape)
        x_a = self.avg_pool(x)
        # print("x_a:", x_a.shape)
        x_m = self.max_pool(x)
        # print("x_m:", x_m.shape)
        # print("1:", x_s.shape)
        # batch_size = x_a.shape[0]
        # x_s = x_s.view(batch_size,18,1)
        # print("2:", x_s.shape)
        x_a = self.fc1_mlp(x_a)
        x_a = F.relu(x_a)
        x_a = self.fc2_mlp(x_a)
        # x_a = F.relu(x_a)
        x_m = self.fc1_mlp(x_m)
        x_m = F.relu(x_m)
        x_m = self.fc2_mlp(x_m)
        # x_m = F.relu(x_m)
        # print("4:", x_s.shape)
        x_s = F.relu(x_a+x_m)
        # x_s = x_s.view(batch_size,18,1,1)
        x = x.mul(x_s)

        x = x.permute(2, 0, 1, 3)
        seq = []
        for x_l in x:
            time_step = x_l.size(0)
            # print("1.0L:", x_l.shape)
            x_l = self.conv1(x_l)
            # print("1.1L:", x_l.shape)
            x_l = self.conv2(x_l)
            # print("1.2L:", x_l.shape)
            x_l = self.conv3(x_l)
            # print("1.3L:", x_l.shape)
            x_l = self.conv4(x_l)
            # print("1.4L:", x_l.shape)
            x_l = x_l.view(time_step, -1)
            # print("1.5L:", x_l.shape)
            seq.append(x_l)
            # print("seq:",len(seq))
        x = torch.stack(seq)
        # print("1.0C:", x_c.shape)
        x = x.permute(1, 0, 2)
        # print("1.1C:", x_c.shape)
        x, _ = self.lstm(x)
        # print("1.2C:", x_c.shape)

        # b, s, h = x.shape
        # x = x[:, s - 1, :]
        # x = x.reshape(b, h)

        batch_shape = x.size(0)
        # print("test1: ", x.shape)
        x = x.contiguous().view(batch_shape, -1)
        # print("1.3C:", x_c.shape)
        #x = self.dropout(x)
        x = self.fc2(F.relu(self.fc1(x)))
        x = F.relu(x)
        # print("1.4C:", x_c.shape)

        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        #x =self.dropout(x)
        x = F.relu(self.fc5(x))

        x = F.softmax(x, dim=1)
        #print("probability:", x)
        return x        


def Net(out_channels1, out_channels2, out_channels3, out_channels4, lstm_hidden, lstm_step, fc1_out, pretrained=False, progress=True):
    """"
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = net_freq60_1d_nopad(out_channels1=out_channels1, out_channels2=out_channels2, out_channels3=out_channels3, out_channels4=out_channels4, lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    # model = net_freq60_1d_nopad_3conv(out_channels1=out_channels1, out_channels2=out_channels2, out_channels3=out_channels3, \
    #                             lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    # model = net_freq60_1d_nopad_2conv(out_channels1=out_channels1, out_channels2=out_channels2, \
    #                             lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    # model = net_freq30_1d_v2_test(lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    return model
