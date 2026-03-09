import os
import glob
import json
import random

def create_split_manifest(root_dir, output_manifest, val_ratio=0.05, seed=42):
    print(f"🌱 Đang quét toàn bộ dữ liệu từ {root_dir}...")
    random.seed(seed)
    
    if not os.path.exists(root_dir):
        print(f"❌ Không tìm thấy thư mục gốc: {root_dir}")
        return
        
    search_pattern = os.path.join(root_dir, "**", "track_*")
    all_tracks = [os.path.normpath(p) for p in glob.glob(search_pattern, recursive=True) if os.path.isdir(p)]
    
    if not all_tracks:
        print(f"❌ LỖI: Không tìm thấy thư mục track_ nào trong {root_dir}!")
        return

    random.shuffle(all_tracks)
    
    split_idx = int(len(all_tracks) * (1 - val_ratio))
    manifest = {
        "train": all_tracks[:split_idx],
        "val": all_tracks[split_idx:]
    }
    
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"✅ Đã gom thành công {len(all_tracks)} tracks từ tất cả các Scenario!")
    print(f"📊 Train: {len(manifest['train'])} | Val: {len(manifest['val'])}")
    print(f"💾 Đã lưu tại: {output_manifest}")

if __name__ == "__main__":
    ROOT_DIR = "./dataset/train"
    MANIFEST_PATH = "./dataset_manifest.json" 
    
    create_split_manifest(ROOT_DIR, MANIFEST_PATH, val_ratio=0.05, seed=42)