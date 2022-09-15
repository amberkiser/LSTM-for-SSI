import torch
import torch.nn as nn


class LSTMNetBasic(nn.Module):
    """
        Basic LSTM net with 1 or more layers

        1: basic 1 LSTM layer, set n_layers = 1
        2: 2 LSTM layers, set n_layers = 2
        7: #1 with gradient clipping
        8: #2 with gradient clipping
    """
    def __init__(self, input_dim, hidden_dim, n_layers, batch_size):
        super(LSTMNetBasic, self).__init__()
        self.output_size = 1
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(self.hidden_dim, self.output_size)  # Fully connected, readout layer

    def init_h0c0(self):
        hidden_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        cell_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        h0c0 = (hidden_0.double(), cell_0.double())
        return h0c0

    def forward(self, x):
        h0c0 = self.init_h0c0()
        x = x.double()
        lstm_out, h0c0 = self.lstm(x, h0c0)
        out = self.fc(lstm_out)
        return out.squeeze()


class LSTMNetLSTMDropout(nn.Module):
    """
        Performs dropout in LSTM, more than 1 LSTM layer required

        3: 2 LSTM layers with 0.5 dropout, set n_layers = 2
        9: #3 with gradient clipping
    """
    def __init__(self, input_dim, hidden_dim, n_layers, batch_size):
        super(LSTMNetLSTMDropout, self).__init__()
        self.output_size = 1
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, dropout=0.5, batch_first=True) # LSTM layer
        self.fc = nn.Linear(self.hidden_dim, self.output_size) # Fully connected, readout layer

    def init_h0c0(self):
        hidden_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        cell_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        h0c0 = (hidden_0.double(), cell_0.double())
        return h0c0

    def forward(self, x):
        h0c0 = self.init_h0c0()
        x = x.double()
        lstm_out, h0c0 = self.lstm(x, h0c0)
        out = self.fc(lstm_out)
        return out.squeeze()


class LSTMNetDropout(nn.Module):
    """
        LSTM with separate dropout layer after LSTM layer(s)

        4: 1 LSTM layer, dropout layer, set n_layers = 1
        5: 2 LSTM layers, dropout layer, set n_layers = 2
        10: #4 with gradient clipping
        11: #5 with gradient clipping
    """
    def __init__(self, input_dim, hidden_dim, n_layers, batch_size):
        super(LSTMNetDropout, self).__init__()
        self.output_size = 1
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first=True) # LSTM layer
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.hidden_dim, self.output_size) # Fully connected, readout layer

    def init_h0c0(self):
        hidden_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        cell_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        h0c0 = (hidden_0.double(), cell_0.double())
        return h0c0

    def forward(self, x):
        h0c0 = self.init_h0c0()
        x = x.double()
        lstm_out, h0c0 = self.lstm(x, h0c0)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out.squeeze()


class LSTMNetDO2(nn.Module):
    """
        LSTM with separate dropout layer after LSTM layers
        Performs dropout in LSTM, more than 1 LSTM layer required

        6: 2 LSTM layers with 0.5 dropout, dropout layer, set n_layers=2
        12: #6 with gradient clipping
    """
    def __init__(self, input_dim, hidden_dim, n_layers, batch_size):
        super(LSTMNetDO2, self).__init__()
        self.output_size = 1
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, dropout=0.5, batch_first=True) # LSTM layer
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.hidden_dim, self.output_size) # Fully connected, readout layer

    def init_h0c0(self):
        hidden_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        cell_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        h0c0 = (hidden_0.double(), cell_0.double())
        return h0c0

    def forward(self, x):
        h0c0 = self.init_h0c0()
        x = x.double()
        lstm_out, h0c0 = self.lstm(x, h0c0)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out.squeeze()
