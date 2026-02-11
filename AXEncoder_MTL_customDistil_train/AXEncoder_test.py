import torch
from transformers import AutoConfig, AutoModel

def check_decoder_compatibility(model_id):
    print(f"🔍 Checking compatibility for: {model_id}")
    
    try:
        # 1. 설정(Config) 로드 및 변조
        # 모델의 설정을 불러와서 "너는 이제부터 디코더야"라고 강제 설정합니다.
        config = AutoConfig.from_pretrained(model_id)
        
        # ★ 핵심 설정 변경
        config.is_decoder = True           # 디코더 모드 활성화 (Masked Self-Attention)
        config.add_cross_attention = True  # 크로스 어텐션 레이어 삽입 요청
        
        print(f"   - Model Type detected: {config.model_type}")
        print("   - Configuration modified (is_decoder=True, add_cross_attention=True)")

        # 2. 모델 인스턴스화 (수술 집도)
        # 변경된 설정을 바탕으로 모델 뼈대를 만듭니다. 
        # 이때 Hugging Face 라이브러리가 자동으로 구조를 변경하려 시도합니다.
        model = AutoModel.from_config(config)
        
        # 3. Cross-Attention 레이어 존재 여부 수색
        # 모델 내부의 모든 레이어 이름을 뒤져서 'cross'나 'CrossAttention'이 있는지 찾습니다.
        has_cross_attention = False
        target_layer_name = ""
        
        for name, module in model.named_modules():
            # 보통 BERT/RoBERTa 계열은 'crossattention'이라는 이름으로 생성됩니다.
            if "crossattention" in name.lower() or "cross_attention" in name.lower():
                has_cross_attention = True
                target_layer_name = name
                break
        
        # 4. 결과 판정
        if has_cross_attention:
            print("\n✅ [SUCCESS] 호환성 확인 완료!")
            print(f"   - Cross-Attention 레이어가 성공적으로 주입되었습니다.")
            print(f"   - 발견된 레이어 예시: {target_layer_name}")
            print("   - 결론: 이 모델은 Encoder-Decoder 구조의 디코더로 사용 가능합니다.")
            return True
        else:
            print("\n❌ [FAILURE] 호환성 확인 실패.")
            print("   - 설정은 변경했으나, Cross-Attention 레이어가 생성되지 않았습니다.")
            print("   - 이 모델 아키텍처는 add_cross_attention 기능을 지원하지 않을 수 있습니다.")
            return False

    except Exception as e:
        print(f"\n❌ [ERROR] 오류 발생: {e}")
        return False

# --- 실행 ---
if __name__ == "__main__":
    # 확인하고자 하는 모델 ID
    TARGET_MODEL = "skt/A.X-Encoder-base"
    
    check_decoder_compatibility(TARGET_MODEL)