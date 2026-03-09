import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Đã đổi lại tên biến thành localization thay vì loc
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        # Đã đổi lại tên biến thành fc_loc thay vì fc
        self.fc_loc = nn.Sequential(nn.Linear(64*16, 128), nn.ReLU(True), nn.Linear(128, 6))
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
        
    def forward(self, x):
        theta = self.fc_loc(self.localization(x).view(-1, 64*16)).view(-1, 2, 3)
        return F.grid_sample(x, F.affine_grid(theta, x.size(), align_corners=False), align_corners=False)

class FrameAttentionFusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        # Đã đổi tên biến thành attention thay vì att
        self.attention = nn.Sequential(
            nn.Conv2d(c, c//2, 1), nn.BatchNorm2d(c//2), nn.ReLU(True), 
            nn.Conv2d(c//2, 1, 1)
        )
    def forward(self, x):
        B, N, C, H, W = x.shape
        scores = F.softmax(self.attention(x.view(B*N, C, H, W)).view(B, N, 1, H, W), dim=1)
        return torch.sum(x * scores, dim=1)

class ResTranOCR_Robust(nn.Module):
    # Giữ nguyên các đối số khởi tạo để không gây lỗi ở file train.py và serve.py
    def __init__(self, vocab_size, feat=256, d_model=256, nhead=8, num_layers=3, dropout=0.3):
        super().__init__()
        self.stn = SpatialTransformerNetwork()
        
        # Khôi phục đúng cấu trúc CNN cũ với 4 Block
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(True), nn.MaxPool2d((4, 1), (4, 1))
        )
        self.fusion = FrameAttentionFusion(feat)
        
        # 🌟 KHU VỰC QUAN TRỌNG NHẤT: Khôi phục mạng LSTM 🌟
        self.lstm = nn.LSTM(feat, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, vocab_size) # Khớp với torch.Size([37, 512]) trong lỗi báo
        
    def forward(self, x):
        B, N, C, H, W = x.shape
        f = self.cnn(self.stn(x.view(B*N, C, H, W)))
        fused = self.fusion(f.view(B, N, f.size(1), f.size(2), f.size(3)))
        
        # Chuyển đổi định dạng tensor phù hợp để đưa vào LSTM
        fused = fused.squeeze(2).permute(0, 2, 1) 
        lstm_out, _ = self.lstm(fused)
        return self.fc(lstm_out)