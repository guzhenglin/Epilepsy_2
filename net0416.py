import torch
import torch.nn as nn
import torch.nn.functional as F

class net_freq60_1d_nopad(nn.Module):
    def __init__(self, out_channels1, out_channels2, out_channels3, out_channels4, lstm_hidden, lstm_step, fc1_out):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out_channels1, kernel_size=5, stride=1, padding=1),
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
                            num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden*lstm_step, fc1_out)    # lstm_hidden*lstm_step
        self.fc2 = nn.Linear(fc1_out, 4)
        self.fc3 = nn.Linear(4*18, 4)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1, x.shape[3])
        # print("1:", x.shape)
        x = x.permute(2, 1, 0, 3, 4)
        # x = torch.squeeze(x)
        # print("2:", x.shape)
        seq_all = []
        for x_c in x:
            seq = []
            for x_l in x_c:
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
            x_c = torch.stack(seq)   
            # print("1.0C:", x_c.shape) 
            x_c = x_c.permute(1, 0, 2)
            # print("1.1C:", x_c.shape) 
            x_c, _ = self.lstm(x_c)
            # print("1.2C:", x_c.shape)

            batch_shape = x_c.size(0)
            x_c = x_c.contiguous().view(batch_shape, -1)
            # print("1.3C:", x_c.shape) 
            x_c = self.fc2(F.relu(self.fc1(self.dropout(x_c))))
            # print("1.4C:", x_c.shape) 
            seq_all.append(x_c)
            # print("seq_all:",len(seq_all))
        x = torch.stack(seq_all)
        # print("1.0X:", x.shape)
        x = x.permute(1,0,2)
        # print("1.1X:", x.shape)
        batch = x.shape[0]
        x = x.contiguous().view(batch, -1)
        # print("1.2X:", x.shape)
        x = self.fc3(F.relu(x))
        # print("1.3X:", x.shape)
        x = F.softmax(x, dim=1)
        #print("probability:", x)
        return x        

class net_freq60_1d_nopad_3conv(nn.Module):
    def __init__(self, out_channels1, out_channels2, out_channels3, lstm_hidden, lstm_step, fc1_out):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=18, out_channels=out_channels1, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(out_channels2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels2, out_channels=out_channels3, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(out_channels3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(in_channels=out_channels3, out_channels=out_channels4, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm1d(out_channels4),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=1)
        # )
        self.lstm = nn.LSTM(input_size=out_channels3, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden*lstm_step, fc1_out)    # lstm_hidden*lstm_step
        self.fc2 = nn.Linear(fc1_out, 4)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3)
        # x = torch.squeeze(x)
        # print("1:", x.shape)
        seq = []
        for x_l in x:
            time_step = x_l.size(0)
            # print("1.0:", x_l.shape)
            x_l = self.conv1(x_l)
            # print("1.1:", x_l.shape)
            x_l = self.conv2(x_l)
            # print("1.2:", x_l.shape)
            x_l = self.conv3(x_l)
            # print("1.3:", x_l.shape)
            # x_l = self.conv4(x_l)
            # print("1.4:", x_l.shape)
            # x_l = self.conv5(x_l)
            # # print("1.5:", x_l.shape)
            # x_l = self.conv6(x_l)
            # # print("1.6:", x_l.shape)
            x_l = x_l.view(time_step, -1)
            # print("1.x:", x_l.shape)
            seq.append(x_l)
        x = torch.stack(seq)
        x = x.permute(1, 0, 2)
        # print("2:", x.shape)
        x, _ = self.lstm(x)
        # print("3:", x.shape)
        batch_shape = x.size(0)
        x = x.contiguous().view(batch_shape, -1)
        # x = x[:, 10736:20736]
        # print("4:", x.shape)
        x = self.fc2(F.relu(self.fc1(self.dropout(x))))
        # x = self.fc1(self.dropout(x))
        # print("5:", x.shape)
        x = F.softmax(x, dim=1)
        #print("probability:", x)
        return x        

class net_freq60_1d_nopad_2conv(nn.Module):
    def __init__(self, out_channels1, out_channels2, lstm_hidden, lstm_step, fc1_out):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=18, out_channels=out_channels1, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=6, stride=6)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(out_channels2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=7)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=out_channels2, out_channels=out_channels3, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm1d(out_channels3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=3, stride=3)
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(in_channels=out_channels3, out_channels=out_channels4, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm1d(out_channels4),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=1)
        # )
        self.lstm = nn.LSTM(input_size=out_channels2, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden*lstm_step, fc1_out)    # lstm_hidden*lstm_step
        self.fc2 = nn.Linear(fc1_out, 4)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3)
        # x = torch.squeeze(x)
        # print("1:", x.shape)
        seq = []
        for x_l in x:
            time_step = x_l.size(0)
            # print("1.0:", x_l.shape)
            x_l = self.conv1(x_l)
            # print("1.1:", x_l.shape)
            x_l = self.conv2(x_l)
            # print("1.2:", x_l.shape)
            # x_l = self.conv3(x_l)
            # print("1.3:", x_l.shape)
            # x_l = self.conv4(x_l)
            # print("1.4:", x_l.shape)
            # x_l = self.conv5(x_l)
            # # print("1.5:", x_l.shape)
            # x_l = self.conv6(x_l)
            # # print("1.6:", x_l.shape)
            x_l = x_l.view(time_step, -1)
            # print("1.x:", x_l.shape)
            seq.append(x_l)
        x = torch.stack(seq)
        x = x.permute(1, 0, 2)
        # print("2:", x.shape)
        x, _ = self.lstm(x)
        # print("3:", x.shape)
        batch_shape = x.size(0)
        x = x.contiguous().view(batch_shape, -1)
        # x = x[:, 10736:20736]
        # print("4:", x.shape)
        x = self.fc2(F.relu(self.fc1(self.dropout(x))))
        # x = self.fc1(self.dropout(x))
        # print("5:", x.shape)
        x = F.softmax(x, dim=1)
        #print("probability:", x)
        return x        

def Net(out_channels1, out_channels2, out_channels3, out_channels4, lstm_hidden, lstm_step, fc1_out, pretrained=False, progress=True):
    """"
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = net_freq60_1d_nopad(out_channels1=out_channels1, out_channels2=out_channels2, out_channels3=out_channels3, out_channels4=out_channels4, \
                                lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    # model = net_freq60_1d_nopad_3conv(out_channels1=out_channels1, out_channels2=out_channels2, out_channels3=out_channels3, \
    #                             lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    # model = net_freq60_1d_nopad_2conv(out_channels1=out_channels1, out_channels2=out_channels2, \
    #                             lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    # model = net_freq30_1d_v2_test(lstm_hidden=lstm_hidden, lstm_step=lstm_step, fc1_out=fc1_out)
    return model
