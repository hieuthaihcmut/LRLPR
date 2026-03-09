import os
import yaml
import json
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models import ResTranOCR_Robust
from src.data import MultiFrameOCRDatasetCached, apply_layout_pattern, clean_text
from src.utils.text_utils import ctc_decode, stoi, vocab_size

def collate(b): 
    return torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b], [x[3] for x in b]

def main():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Đã tắt W&B để in log thẳng ra terminal
    wandb.init(project=config['project']['name'], config=config, mode="disabled")
    
    config_device = config['project'].get('device', 'auto')
    if config_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config_device)
        
    print(f"🚀 Đang chạy pipeline trên: {device}")
    
    with open(config['data']['manifest_path'], "r", encoding="utf-8") as f:
        data_splits = json.load(f)
    tr_tracks, va_tracks = data_splits["train"], data_splits["val"]
    
    print(f"📦 Dữ liệu: {len(tr_tracks)} Train | {len(va_tracks)} Val")

    tr_dataset = MultiFrameOCRDatasetCached(tr_tracks, config['project']['cache_path'], th=config['data']['img_h'], tw=config['data']['img_w'], is_train=True)
    va_dataset = MultiFrameOCRDatasetCached(va_tracks, config['project']['cache_path'], th=config['data']['img_h'], tw=config['data']['img_w'], is_train=False)
    
    tr_loader = DataLoader(tr_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=2, collate_fn=collate)
    va_loader = DataLoader(va_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=2, collate_fn=collate)

    model = ResTranOCR_Robust(vocab_size, d_model=config['model']['d_model'], nhead=config['model']['nhead'], num_layers=config['model']['num_layers']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config['train']['max_lr'], steps_per_epoch=len(tr_loader), epochs=config['train']['epochs'], pct_start=0.2)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_acc = 0.0
    os.makedirs(config['project']['work_dir'], exist_ok=True)
    best_path = os.path.join(config['project']['work_dir'], "best_ocr_model.pth")
    
    for ep in range(1, config['train']['epochs'] + 1):
        model.train()
        tl = 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}/{config['train']['epochs']}")
        
        for X, txts, lays, _ in pbar:
            logits = model(X.to(device))
            targs, tlens = [], []
            for t in txts:
                seq = [stoi[c] for c in t if c in stoi]
                targs.extend(seq); tlens.append(len(seq))
                
            logp = logits.log_softmax(-1).permute(1,0,2)
            loss = ctc(logp, torch.tensor(targs).to(device), torch.full((logp.size(1),), logp.size(0)), torch.tensor(tlens).to(device))
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
            
            tl += loss.item()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, txts, lays, _ in va_loader:
                preds = ctc_decode(model(X.to(device)))
                for p, gt, lay in zip(preds, txts, lays):
                    if apply_layout_pattern(p, lay) == clean_text(gt):
                        correct += 1
                    total += 1
                    
        val_acc = correct / max(1, total)
        avg_train_loss = tl / len(tr_loader)
        
        print(f"✅ Ep {ep}: Loss={avg_train_loss:.4f} | VAL ACC={val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(" 💾 Đã lưu model xịn nhất!")

if __name__ == "__main__":
    main()