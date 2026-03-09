import os
import yaml
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from typing import List
import uvicorn

from src.models import TinyCornerNet, CornerPredictor, ResTranOCR_Robust
from src.data.preprocess import warp_plate
from src.utils.text_utils import ctc_decode, vocab_size, itos

app = FastAPI(title="LPR Multi-frame OCR API")

# --- KHỞI TẠO BIẾN GLOBAL ĐỂ LOAD MODEL 1 LẦN DUY NHẤT KHI START SERVER ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
c_pred = None
o_model = None

@app.on_event("startup")
def load_models():
    global c_pred, o_model
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load Corner Net
    c_model = TinyCornerNet().to(device)
    c_model.load_state_dict(torch.load(config['project']['corner_ckpt'], map_location=device))
    c_pred = CornerPredictor(c_model, device)
    
    # Load OCR Net
    o_model = ResTranOCR_Robust(vocab_size).to(device)
    best_ocr_path = os.path.join(config['project']['work_dir'], "best_ocr_model.pth")
    o_model.load_state_dict(torch.load(best_ocr_path, map_location=device))
    o_model.eval()
    print("✅ Đã load models thành công!")

@app.post("/predict")
async def predict_plate(files: List[UploadFile] = File(...)):
    """ Nhận vào danh sách ảnh (từ 1 đến 5 frames) và trả về biển số xe """
    frames = []
    
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            frames.append(torch.zeros(3, 32, 128))
            continue
            
        # Tiền xử lý giống hệt file notebook gốc
        c = c_pred.predict_corners(img)
        rgb = (cv2.resize(warp_plate(img, c, 32), (128, 32))[:,:,::-1].astype(np.float32)/255.0 - 0.5)/0.5
        frames.append(torch.from_numpy(rgb).permute(2,0,1))
        
    # Đệm thêm cho đủ 5 frames nếu client gửi thiếu
    while len(frames) < 5:
        frames.append(torch.zeros(3, 32, 128))
    frames = frames[:5]
        
    X = torch.stack(frames, dim=0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = o_model(X)
        probs = torch.softmax(logits, dim=-1)
        idx, conf = probs.argmax(-1)[0], probs.max(-1).values[0]
        
        out, out_c, prev = [], [], None
        for t in range(idx.size(0)):
            c = int(idx[t])
            if c != 0 and c != prev: 
                out.append(itos[c])
                out_c.append(float(conf[t]))
            prev = c
            
        raw_p = "".join(out)
        avg_conf = sum(out_c) / max(1, len(out_c)) if out_c else 0.0
        
    return {"plate": raw_p, "confidence": round(avg_conf, 4)}

if __name__ == "__main__":
    # Khởi chạy server ở port 8000
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)