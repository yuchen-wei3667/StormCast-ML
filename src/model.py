import torch
import torch.nn as nn

class StormCellLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=2):
        super(StormCellLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def load_trained_weights(self, weights_path='trained_storm_lstm.pth'):
        """Load trained weights from file"""
        try:
            self.load_state_dict(torch.load(weights_path))
            return True
        except FileNotFoundError:
            return False

    def predict_from_json(self, json_input, use_trained=True):
        """
        Takes a single storm cell JSON object (dict) and predicts the motion vector.
        If use_trained=True, will attempt to load trained weights first.
        """
        from src.data_loader import parse_storm_cell_json
        
        # Try to load trained weights
        if use_trained:
            if self.load_trained_weights():
                pass  # Weights loaded successfully
        
        # Preprocess input
        input_tensor = parse_storm_cell_json(json_input)
        
        if input_tensor.size(1) == 0:
            return None # No history to predict from
            
        # Run inference
        self.eval()
        with torch.no_grad():
            prediction = self(input_tensor)
            
        return prediction.numpy()[0] # Return as numpy array [vx, vy]
