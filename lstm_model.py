import torch
import torch.nn as nn

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.hidden_dim_1 = 150
        self.hidden_dim_2 = 50

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 = nn.Linear(hidden_size, self.hidden_dim_1) #fully connected 1
        self.fc_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2) #fully connected 2
        self.fc_3 = nn.Linear(self.hidden_dim_2, num_classes) #fully connected last layer


        self.relu = nn.LeakyReLU()
    
    def forward(self,x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc_2(out) #second Dense
        out = self.relu(out) #relu
        out = self.fc_3(out) #Final Output

        output = self.relu(output)
        output = self.fc_1(output)
        output = self.relu(output)
        output = self.fc_2(output)
        output = self.relu(output)
        output = self.fc_3(output)

        return output, out
    
    def step(self, x, labels, args):
        pred_seq, pred = self.forward(x)
        #loss = nn.functional.mse_loss(pred, labels, reduction='sum')
        loss = torch.sqrt(torch.sum(torch.square(pred_seq - labels)))
        
        return loss, None, None