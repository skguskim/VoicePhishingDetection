# train_student.py
import os
import torch
import csv  # [★추가] CSV 저장을 위해
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import CONFIG
from model import DistillableRoBERTaModel, initialize_student_weights
from dataset import VoicePhishingDataset
from trainer import DistillationTrainer
# [★수정] get_logger, save_checkpoint 임포트 추가
from utils import set_seed, prepare_data, get_logger, save_checkpoint 

def main():
    set_seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 로거 및 CSV 초기화
    logger = get_logger(CONFIG['OUTPUT_DIR'])
    logger.info(f"--- [Student Training] Device: {device} ---")
    
    # 로그 파일(CSV) 생성 (헤더 작성)
    log_csv_path = os.path.join(CONFIG['OUTPUT_DIR'], 'training_log.csv')
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])
    
    # 2. 데이터 준비
    train_path, test_path = prepare_data(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    
    train_dataset = VoicePhishingDataset(train_path, tokenizer, inference_mode=False)
    test_dataset = VoicePhishingDataset(test_path, tokenizer, inference_mode=True) # 검증용
    
    # GPU 사용 시 pin_memory=True 설정을 통해 데이터 전송 속도 향상
    # Windows 사용자는 num_workers=0으로 설정
    # Linux나 Colab 사용자는 num_workers를 2 또는 4로 설정
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, **kwargs)
    
    # 3. 모델 초기화
    logger.info("Initializing Models...")
    
    teacher_model = DistillableRoBERTaModel(CONFIG['MODEL_NAME'], CONFIG['NUM_CLASSES'], is_student=False)
    teacher_path = os.path.join(CONFIG['OUTPUT_DIR'], "teacher_best.pt")
    
    if os.path.exists(teacher_path):
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
    else:
        raise FileNotFoundError("Teacher weights not found!")
    
    student_model = DistillableRoBERTaModel(CONFIG['MODEL_NAME'], CONFIG['NUM_CLASSES'], is_student=True, student_layer_num=CONFIG['STUDENT_LAYER'])
    
    initialize_student_weights(teacher_model, student_model)
    
    # 4. 학습 시작
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataloader=train_loader,
        device=device,
        lr=CONFIG['LEARNING_RATE']
    )
    
    best_acc = 0.0
    
    logger.info("Start Training...")
    
    for epoch in range(CONFIG['EPOCHS']):
        # 1) 학습
        train_loss = trainer.train_epoch(epoch)
        
        # 2) 검증
        val_loss, val_acc = trainer.evaluate(test_loader)
        
        # 3) 로그 기록 (콘솔 + CSV)
        logger.info(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        with open(log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss, val_acc])

        # 4) 체크포인트 저장
        # [Last] 언제든 재개할 수 있도록 매 에폭마다 덮어쓰기 (Optimizer 상태 포함)
        save_checkpoint(
            student_model, trainer.optimizer, epoch, train_loss,
            os.path.join(CONFIG['OUTPUT_DIR'], "student_last.pt")
        )

        # [Best] 최고 성능일 때만 가중치 저장 (Model Weight Only)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(CONFIG['OUTPUT_DIR'], "student_best.pt")
            torch.save(student_model.state_dict(), best_path)
            logger.info(f"★ New Best Accuracy! Model saved to {best_path}")

    logger.info(f"All Finished. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()