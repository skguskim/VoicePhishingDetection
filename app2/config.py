import os
import torch

# [0. 절대 경로 설정]
# 현재 파일(config.py)이 위치한 폴더를 기준으로 모든 경로를 설정합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # [1. 기본 환경 설정]
    'PROJECT_NAME': "ModernBERT_Distillation_Multilingual",
    'SEED': 42,
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    
    # [중요] Windows에서 KoEDA(Java) 충돌 방지를 위해 전처리 시에는 0 권장.
    # 전처리 완료 후(.pt 파일 생성 후)에는 4로 올려도 무방합니다.
    'NUM_WORKERS': 4,
    
    # [2. 경로 설정] (절대 경로 적용)
    # dataset_master.csv가 프로젝트 상위 폴더에 있다고 가정 (필요시 수정)
    'MASTER_DATA_PATH': os.path.join(os.path.dirname(BASE_DIR), 'dataset_master.csv'),
    'PROCESSED_DATA_DIR': os.path.join(BASE_DIR, 'processed_data'),
    
    # 결과 저장 경로
    'OUTPUT_DIR_TEACHER': os.path.join(BASE_DIR, 'ModernBERT_Result', 'Teacher'),
    'OUTPUT_DIR_TA': os.path.join(BASE_DIR, 'ModernBERT_Result', 'TA'),
    'OUTPUT_DIR_STUDENT': os.path.join(BASE_DIR, 'ModernBERT_Result', 'Student'),
    
    # [3. 모델 기본 설정]
    'BASE_MODEL_NAME': "neavo/modern_bert_multilingual", 
    'NUM_LABELS': 2,
    'MAX_LENGTH': 1024,
    
    # [4. 학습 전략: Gradient Accumulation]
    # 메모리 부족(OOM) 방지를 위해 작은 배치로 여러 번 나누어 업데이트합니다.
    # Teacher: 8(Batch) * 4(Accum) = 32(Effective Batch)
    'ACCUMULATION_STEPS': 4, 
    
    # [5. 모델 아키텍처 및 학습 스케줄]
    
    # (A) Teacher: Fine-tuning
    'TEACHER': {
        'config': {
            'hidden_size': 768,
            'num_hidden_layers': 22,
            'num_attention_heads': 12,
            'intermediate_size': 1152
        },
        'train': {
            'epochs': 5,
            'lr': 3e-5, 
            'batch_size': 8 # 실제 GPU 메모리에 올라가는 크기 (작게 설정)
        }
    },

    # (B) TA (Teacher Assistant): Width Pruning
    # 코드 호환성을 위해 키 이름을 'TA_CONFIG' 형태로도 접근 가능하게 매핑 필요할 수 있음
    'TA': {
        'config': {
            'hidden_size': 384,      
            'num_hidden_layers': 22, 
            'num_attention_heads': 6,
            'intermediate_size': 576
        },
        'train': {
            'epochs': 5,
            'lr': 1e-4,
            'batch_size': 16 # TA도 메모리 상황에 따라 조절 (Accumulation 적용 권장)
        }
    },

    # (C) Student: Depth Pruning + Recovery
    'STUDENT': {
        'config': {
            'hidden_size': 384,
            'num_hidden_layers': 11,
            'num_attention_heads': 6,
            'intermediate_size': 576
        },
        'train': {
            'epochs': 10, 
            'lr': 5e-4,   
            'batch_size': 32 # 모델이 작으므로 배치를 키울 수 있음
        }
    },
    
    # [6. Optimizer]
    'OPTIMIZER': {
        'type': 'AdamW',
        'betas': (0.9, 0.98),
        'eps': 1e-6,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1 
    },

    # [7. Scientific Initialization]
    'INIT_METHOD': {
        'layer_selection': 'bi_score',
        'width_selection': 'importance',
        'pruning_ratio': 0.5
    },

    # [8. Distillation Loss]
    'DISTILLATION': {
        'TEMPERATURE': 2.0,
        'ALPHA': 0.0, 
        'HIDDEN_LOSS_TYPE': 'mse', 
        'USE_PROJECTION': True    
    },
    
    # [9. 데이터 처리 하이퍼파라미터]
    'WINDOW_SIZE': 15,
    'STRIDE': 5,
    'AUG_PROB': 0.5,
    'USE_AMP': True
}

# [편의성을 위한 별칭 매핑]
# architecture.py 등에서 CONFIG['TA_CONFIG']로 접근하므로 매핑 추가
CONFIG['TA_CONFIG'] = CONFIG['TA']['config']
CONFIG['STUDENT_CONFIG'] = CONFIG['STUDENT']['config']
