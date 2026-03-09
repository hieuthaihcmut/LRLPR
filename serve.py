import os
import yaml
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from typing import List
import uvicorn
from contextlib import asynccontextmanager

from src.models import TinyCornerNet, CornerPredictor, ResTranOCR_Robust
from src.data.preprocess import warp_plate, apply_layout_pattern
from src.utils.text_utils import ctc_decode, vocab_size, itos

# Biến toàn cục
c_pred = None
o_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global c_pred, o_model
    with open("configs/default.yaml", "r", encoding="utf-8") as f: 
        config = yaml.safe_load(f)
    
    print("\n⏳ Đang nạp mô hình...")
    
    # 1. NẠP MÔ HÌNH NẮN GÓC (CORNER NET)
    c_model = TinyCornerNet().to(device)
    corner_path = config['project']['corner_ckpt']
    
    if os.path.exists(corner_path):
        c_model.load_state_dict(torch.load(corner_path, map_location=device))
        print(f"✅ [1/2] Đã nạp CornerNet từ: {corner_path}")
    else:
        print(f"❌ CẢNH BÁO: KHÔNG TÌM THẤY {corner_path}!")
        print("   -> AI sẽ bị 'mù' mạng nắn góc, cắt ảnh sai và đọc ra chuỗi rỗng!")
        
    c_pred = CornerPredictor(c_model, device)
    
    # 2. NẠP MÔ HÌNH OCR (BẢN LSTM - feat=512)
    o_model = ResTranOCR_Robust(vocab_size, feat=512).to(device)
    best_path = os.path.join(config['project']['work_dir'], "best_ocr_model.pth")
    
    if os.path.exists(best_path):
        o_model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"✅ [2/2] Đã nạp OCR Model từ: {best_path}")
    else:
        print(f"❌ CẢNH BÁO: KHÔNG TÌM THẤY {best_path}!")
        
    o_model.eval()
    print("🚀 HỆ THỐNG API ĐÃ SẴN SÀNG NHẬN ẢNH!\n")
    
    yield # Trả quyền điều khiển cho FastAPI chạy
    
    print("🛑 Đang tắt hệ thống...")

# Khởi tạo App với lifespan mới
app = FastAPI(title="LPR OCR System API", lifespan=lifespan)

@app.post("/predict")
async def predict_plate(layout: str = "Mercosur", files: List[UploadFile] = File(...)):
    """API Nhận 1 đến 5 ảnh (các frame của 1 xe) và trả về biển số"""
    frames = []
    
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            frames.append(torch.zeros(3, 32, 128))
            continue
            
        # Nắn góc và tiền xử lý
        c = c_pred.predict_corners(img)
        rgb = (cv2.resize(warp_plate(img, c, 32), (128, 32))[:,:,::-1].astype(np.float32)/255.0 - 0.5)/0.5
        
        # 🕵️ LƯU ẢNH DEBUG ĐỂ BẮT BỆNH
        debug_img = (rgb * 0.5 + 0.5) * 255.0
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        cv2.imwrite(f"debug_{safe_filename}.jpg", debug_img[:,:,::-1])
        
        frames.append(torch.from_numpy(rgb).permute(2,0,1))
        
    # Đệm cho đủ 5 frame nếu thiếu
    # Nhân bản frame cuối cùng cho đủ 5 frames thay vì nhét ảnh đen
    if not frames:
        return {"plate": "", "confidence": 0.0}
        
    while len(frames) < 5: 
        frames.append(frames[-1])
    frames = frames[:5]
        
    X = torch.stack(frames, dim=0).unsqueeze(0).to(device)
    
    # Dự đoán
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
        final_p = apply_layout_pattern(raw_p, layout)
        
    return {
        "plate": final_p, 
        "raw_text": raw_p,
        "confidence": round(avg_conf, 4)
    }

if __name__ == "__main__":
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=True)