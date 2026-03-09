import torch
import torch.nn as nn
import cv2
import numpy as np

ORDER = ["top-left","top-right","bottom-right","bottom-left"]

class TinyCornerNet(nn.Module):
    def __init__(self):
        super().__init__()
        block = lambda cin, cout: nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(True), nn.MaxPool2d(2))
        self.backbone = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, 8), nn.Sigmoid())
    def forward(self, x): return self.head(self.backbone(x))

class CornerPredictor:
    def __init__(self, model, device, in_size=(64,32)):
        self.model = model.eval()
        self.device = device
        self.wt, self.ht = in_size[0], in_size[1]
    @torch.no_grad()
    def predict_corners(self, img_bgr):
        h, w = img_bgr.shape[:2]
        rgb = (cv2.resize(img_bgr, (self.wt, self.ht))[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        c = self.model(x).squeeze(0).cpu().numpy().astype(np.float32).reshape(4,2)
        c[:,0] *= float(w); c[:,1] *= float(h)
        return {ORDER[i]: [float(c[i,0]), float(c[i,1])] for i in range(4)}