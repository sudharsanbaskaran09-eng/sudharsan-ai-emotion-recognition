import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """
    Spatial Attention mechanism mapping specific potential local dominant regions 
    (lips, eyes, mouth)
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # Using 1x1 convolution to identify spatial feature importance
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attn_weights = torch.sigmoid(self.conv(x))
        return x * attn_weights

class ChannelAttention(nn.Module):
    """
    Channel Attention capturing global information constraints
    (overall face structure, lighting, poses)
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared Multilayer Perceptron
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn_weights = self.sigmoid(avg_out + max_out)
        return x * attn_weights

class DualAttentionCrossFusion(nn.Module):
    """
    Combines both Spatial and Channel attentions in parallel 
    and uses cross-fusion to merge them avoiding information destruction.
    """
    def __init__(self, in_channels):
        super(DualAttentionCrossFusion, self).__init__()
        self.spatial = SpatialAttention(in_channels)
        self.channel = ChannelAttention(in_channels)
        
        # DCNN cross-conv to seamlessly fuse sizes natively
        self.cross_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
    def forward(self, x):
        s_attn = self.spatial(x)
        c_attn = self.channel(x)
        
        # Cross-fusion: concatenate maps
        fused = torch.cat([s_attn, c_attn], dim=1)
        return self.cross_conv(fused)

class DCNN(nn.Module):
    """
    Deep Convolutional Neural Network module matching Requirement Form:
    - 3 Conv layers
    - 2 MaxPool layers
    """
    def __init__(self):
        super(DCNN, self).__init__()
        # Expected input: 1 Channel (HOG preprocessed grayscale), 64x64
        self.c1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.s1 = nn.MaxPool2d(2, 2)
        
        self.c2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.s2 = nn.MaxPool2d(2, 2)
        
        self.c3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.s1(self.relu(self.c1(x))) # Output: 32x32
        x = self.s2(self.relu(self.c2(x))) # Output: 16x16
        x = self.relu(self.c3(x))          # Output: 16x16
        return x

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM module processing both Forward/Backward time series
    for continuous facial representations.
    """
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
                            
    def forward(self, x):
        # Output shape: (Batch, Sequence Length, Hidden_Dim * 2)
        out, _ = self.lstm(x)
        return out

class DCNN_BiLSTM_DAM(nn.Module):
    """
    Full Architecture integration exactly matching the college requirement constraints.
    """
    def __init__(self, num_classes=7):
        super(DCNN_BiLSTM_DAM, self).__init__()
        
        self.dcnn = DCNN()
        self.dam = DualAttentionCrossFusion(in_channels=128)
        
        # After 16x16 pooling from DCNN -> spatial flattened length = 256
        self.seq_len = 16 * 16 
        self.feature_channels = 128
        
        # Bi-LSTM for sequence temporal features
        self.bilstm = BiLSTM(input_dim=self.feature_channels, hidden_dim=64)
        
        # Fully connected block
        bilstm_output_features = 64 * 2 # Bidirectional
        self.fc1 = nn.Linear(bilstm_output_features * self.seq_len, 300)
        self.dropout = nn.Dropout(0.4)
        # Softmax classifier implemented automatically by PyTorch CrossEntropyLoss on the final un-activated layer
        self.fc2 = nn.Linear(300, num_classes)
        
    def forward(self, x):
        # 1. Feature Extraction (DCNN)
        features = self.dcnn(x)                       
        
        # 2. Attention Focus (Dual Attention Mechanism)
        attention_maps = self.dam(features)           
        
        # Reshaping Spatial Dims into sequences for BiLSTM
        B, C, H, W = attention_maps.size()
        seq_input = attention_maps.view(B, C, H * W).permute(0, 2, 1) 
        
        # 3. Sequential Processing (Bi-LSTM)
        bilstm_out = self.bilstm(seq_input)           
        
        # Flatten for classification
        flat_out = bilstm_out.reshape(B, -1)           
        
        # 4. Classification
        fc1_out = self.fc1(flat_out)                  
        dropped_out = self.dropout(fc1_out)
        
        # Raw logits for external Softmax
        output = self.fc2(dropped_out)                    
        
        return output

# --- Example Usage ---
if __name__ == "__main__":
    model = DCNN_BiLSTM_DAM(num_classes=7)
    dummy_input = torch.randn(1, 1, 64, 64) 
    predictions = model(dummy_input)
    print(f"Model Output Shape (Batches, Classes): {predictions.shape}")
