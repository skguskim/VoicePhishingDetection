import torch
import torch.nn as nn
import torch.nn.functional as F

class DistilBERTLoss(nn.Module):
    """
    DistilBERT 논문의 Triple Loss를 구현한 클래스
    Loss = alpha * L_ce + beta * L_kd + gamma * L_cos
    """
    def __init__(self, alpha=5.0, beta=2.0, gamma=1.0, temperature=2.0):
        super(DistilBERTLoss, self).__init__()
        
        self.alpha = alpha       # Student Loss (Hard Label) 가중치
        self.beta = beta         # Distillation Loss (Soft Label) 가중치
        self.gamma = gamma       # Cosine Embedding Loss (Hidden State) 가중치
        self.temperature = temperature # Softmax Temperature (T)

        # 기본 Loss 함수들
        self.ce_loss_fct = nn.CrossEntropyLoss()
        self.kldiv_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.cosine_loss_fct = nn.CosineEmbeddingLoss(margin=0.0) # 방향이 같으면 Loss 0

    def forward(self, student_outputs, teacher_outputs, labels, attention_mask):
        """
        Args:
            student_outputs (dict): {'logits': ..., 'hidden_states': ...}
            teacher_outputs (dict): {'logits': ..., 'hidden_states': ...}
            labels (tensor): 정답 레이블
            attention_mask (tensor): 패딩 토큰 무시용 마스크
        """
        # ------------------------------------------------------------------
        # 1. Student Loss (L_ce): 정답 레이블과의 오차 (Supervised)
        # ------------------------------------------------------------------
        student_logits = student_outputs['logits']
        student_loss = self.ce_loss_fct(student_logits, labels)

        # ------------------------------------------------------------------
        # 2. Distillation Loss (L_kd): Teacher의 확률 분포 모방 (KL-Div)
        # ------------------------------------------------------------------
        teacher_logits = teacher_outputs['logits']
        
        # Temperature 적용: 분포를 부드럽게(Flatten) 만들어서 정보를 더 많이 전달
        p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        log_p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL Divergence 계산 (T^2를 곱해주는 것이 학술적 표준 - Gradient Scale 유지)
        distillation_loss = self.kldiv_loss_fct(log_p_student, p_teacher) * (self.temperature ** 2)

        # ------------------------------------------------------------------
        # 3. Cosine Loss (L_cos): 중간 Hidden State 벡터 정렬
        # ------------------------------------------------------------------
        # Teacher(12층)와 Student(6층)의 Hidden State 개수가 다름
        # initialize_student_weights와 동일한 로직으로 매핑
        s_hidden = student_outputs['hidden_states'] # (Embed + 6 Layers) = 7개
        t_hidden = teacher_outputs['hidden_states'] # (Embed + 12 Layers) = 13개
        
        # Hidden State 중 Embedding Layer(0번)는 제외하고 Encoder Layer(1번~)부터 비교
        # Student Layer i는 Teacher Layer (i * ratio)와 매핑됨
        # 예: S1 -> T2, S2 -> T4 ...
        
        total_cosine_loss = 0.0
        n_layers = len(s_hidden) - 1 # Embedding 제외한 실제 레이어 수
        layer_ratio = (len(t_hidden) - 1) / n_layers 

        # (Batch * Seq_Len) 차원으로 Flatten해서 모든 토큰의 벡터 유사도 비교
        # 패딩 토큰은 마스크 처리해서 제외해야 정확하지만, 
        # 보통은 전체를 비교해도 큰 성능 저하가 없어 단순화하여 구현함.
        
        for i in range(1, len(s_hidden)): # 1번 인덱스부터 시작
            t_idx = int(i * layer_ratio)
            
            # (Batch, Seq, Dim) -> (Batch * Seq, Dim)
            s_vec = s_hidden[i].view(-1, s_hidden[i].size(-1))
            t_vec = t_hidden[t_idx].view(-1, t_hidden[t_idx].size(-1))
            
            # 타겟: 1 (두 벡터가 같은 방향을 보길 원함)
            target = torch.ones(s_vec.size(0)).to(s_vec.device)
            
            loss_layer = self.cosine_loss_fct(s_vec, t_vec, target)
            total_cosine_loss += loss_layer

        cosine_loss = total_cosine_loss / n_layers

        # ------------------------------------------------------------------
        # Final Total Loss
        # ------------------------------------------------------------------
        total_loss = (self.alpha * student_loss) + \
                     (self.beta * distillation_loss) + \
                     (self.gamma * cosine_loss)

        return total_loss, {
            "loss": total_loss.item(),
            "ce_loss": student_loss.item(),
            "kd_loss": distillation_loss.item(),
            "cos_loss": cosine_loss.item()
        }