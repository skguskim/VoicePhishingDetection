import os
import csv
import glob
from pathlib import Path

# Config
# Assume script is in AXEncoder_MTL_customDistil_train
# BASE_DIR = AXEncoder_MTL_customDistil_train
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = Parent of BASE_DIR (i.e. /home/j2hoon10)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
# DATA_ROOT = /home/j2hoon10/data/preprocessing
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "preprocessing")

def verify_coverage():
    print(f"🔍 Verifying Data Coverage...")
    print(f"   Base Dir: {BASE_DIR}")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Data Root: {DATA_ROOT}")
    
    # 1. Load CSVs
    train_csv_path = os.path.join(BASE_DIR, "data", "train.csv")
    val_csv_path = os.path.join(BASE_DIR, "data", "val.csv")
    
    if not os.path.exists(train_csv_path):
        print(f"❌ Train CSV not found: {train_csv_path}")
        return
    if not os.path.exists(val_csv_path):
        print(f"❌ Val CSV not found: {val_csv_path}")
        return
        
    # 2. Extract Expected Files from CSVs
    expected_files = set()
    rows_count = 0
    
    for csv_path, name in [(train_csv_path, "Train"), (val_csv_path, "Val")]:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_count += 1
                category = row['category']
                raw_path = row['json_path']
                filename = os.path.basename(raw_path.replace("\\", "/"))
                
                # Construct relative path from DATA_ROOT for comparison
                # e.g., "category/filename.json"
                rel_path = os.path.join(category, filename)
                expected_files.add(rel_path)
                
    print(f"   Processed {rows_count} rows from CSVs")
    print(f"   Total Unique Files in CSVs: {len(expected_files)}")
    
    # 3. Scan Actual Files on Disk
    actual_files = set()
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Data Root not found: {DATA_ROOT}")
        return
        
    print(f"   Scanning {DATA_ROOT}...")
    # Walk through DATA_ROOT
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if not file.endswith('.json'):
                continue
                
            abs_path = os.path.join(root, file)
            # Get relative path from DATA_ROOT
            rel_path = os.path.relpath(abs_path, DATA_ROOT)
            actual_files.add(rel_path)
            
    print(f"   Total JSON Files on Disk: {len(actual_files)}")
    
    # 4. Compare
    missing_on_disk = expected_files - actual_files
    unused_on_disk = actual_files - expected_files
    
    print("\n   [Results]")
    print(f"   Matched Files: {len(expected_files & actual_files)}")
    
    if missing_on_disk:
        print(f"   ❌ Missing Files (In CSV but NOT on disk): {len(missing_on_disk)}")
        # Print first 5
        for f in list(missing_on_disk)[:5]:
            print(f"      - {f}")
        if len(missing_on_disk) > 5:
            print("      ... and more")
            
    else:
        print("   ✅ All CSV entries exist on disk.")
        
    if unused_on_disk:
        print(f"   ⚠️ Unused Files (On disk but NOT in CSV): {len(unused_on_disk)}")
        # Print first 5
        for f in list(unused_on_disk)[:5]:
            print(f"      - {f}")
        if len(unused_on_disk) > 5:
            print("      ... and more")
    else:
        print("   ✅ All disk files are used in CSVs.")
        
    coverage = len(expected_files & actual_files) / len(actual_files) if len(actual_files) > 0 else 0
    print(f"\n   📈 Data Coverage: {coverage*100:.2f}% of disk files are used.")

if __name__ == "__main__":
    verify_coverage()
