import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(nn.Linear(64*16, 128), nn.ReLU(True), nn.Linear(128, 6))
        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
    def forward(self, x):
        theta = self.fc(self.loc(x).view(-1, 64*16)).view(-1, 2, 3)
        return F.grid_sample(x, F.affine_grid(theta, x.size(), align_corners=False), align_corners=False)

class ResNetOCR(nn.Module):
    def __init__(self, feat=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(True), nn.MaxPool2d((4, 1), (4, 1)),
        )
    def forward(self, x): return self.net(x)

class FrameAttentionFusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(nn.Conv2d(c, c//2, 1), nn.BatchNorm2d(c//2), nn.ReLU(True), nn.Conv2d(c//2, 1, 1))
    def forward(self, x):
        B, N, C, H, W = x.shape
        scores = F.softmax(self.att(x.view(B*N, C, H, W)).view(B, N, 1, H, W), dim=1)
        return torch.sum(x * scores, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class ResTranOCR_Robust(nn.Module):
    def __init__(self, vocab_size, feat=256, d_model=256, nhead=8, num_layers=3, dropout=0.3):
        super().__init__()
        self.stn, self.cnn, self.fusion = SpatialTransformerNetwork(), ResNetOCR(feat), FrameAttentionFusion(feat)
        self.proj, self.pos_encoder = nn.Linear(feat, d_model), PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_model*4, batch_first=True, dropout=dropout), num_layers)
        self.dropout, self.fc = nn.Dropout(dropout), nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, N, C, H, W = x.shape
        f = self.cnn(self.stn(x.view(B*N, C, H, W)))
        fused = self.fusion(f.view(B, N, f.size(1), f.size(2), f.size(3)))
        return self.fc(self.dropout(self.transformer(self.pos_encoder(self.proj(fused.squeeze(2).permute(0, 2, 1))))))