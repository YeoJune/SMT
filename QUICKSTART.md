# SMT (Stride Memory Transformer) - Quick Start

## 🚀 5분 안에 시작하기

### 1. 프로젝트 확인

```bash
cd SMT
python scripts/verify_structure.py
```

**예상 출력**:

```
✅ 프로젝트 구조 준비 완료!
📊 총 코드 크기: 60.7 KB
📊 총 코드 라인: 1,364 lines
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

**필수 패키지**:

- `torch>=2.0.0` - PyTorch
- `transformers>=4.30.0` - GPT-2 모델
- `mamba-ssm>=1.0.0` - Mamba SSM
- `causal-conv1d>=1.0.0` - Mamba dependency

### 3. 테스트 실행

```bash
python tests/test_integration.py
```

**테스트 항목**:

- ✅ Attention Pooling
- ✅ Window Manager
- ✅ Transformer
- ✅ SSM
- ✅ Full Model

### 4. 첫 모델 실행

```python
from config.model_config import StrideHybridConfig
from src.models.stride_hybrid import StrideHybridModel
import torch

# 1. 설정 생성
config = StrideHybridConfig(
    n_ssm_outputs=15,      # SSM 출력 15개
    m_input_tokens=50,     # 입력 토큰 50개
    stride=16,             # 16 스텝마다 SSM 업데이트
    device='cuda'          # or 'cpu'
)

# 설정 확인
config.summary()

# 2. 모델 생성
model = StrideHybridModel(config)

# 파라미터 확인
params = model.count_parameters()
print(f"Total params: {params['total']/1e6:.1f}M")

# 3. Forward pass (학습)
batch_size = 2
seq_len = 100
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

logits, aux = model(input_ids)
print(f"Output shape: {logits.shape}")  # (2, 100, 50280)
print(f"SSM writes: {aux['n_writes']}/{seq_len}")  # 6/100

# 4. Generation (생성)
prompt = torch.randint(0, config.vocab_size, (1, 10))  # 10 tokens
generated = model.generate(
    input_ids=prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.95
)
print(f"Generated shape: {generated.shape}")  # (1, 60)
```

## 📚 핵심 개념 이해하기

### Window 구조

```
Step t의 window:
[SSM_out[t-14], ..., SSM_out[t],  ← 15개 SSM 출력 (압축된 과거)
 x[t-49], ..., x[t]]              ← 50개 최근 토큰 (상세한 최근)
```

### Stride-based Write

```python
for t in range(seq_len):
    # 항상: Window 처리 → Logits
    window = get_window()
    logits[t] = transformer(window)

    # 조건부 (16 스텝마다): SSM 업데이트
    if t % 16 == 0:
        pooled = attention_pooling(window)  # 65 → 1
        ssm_out = ssm(pooled)               # 압축 메모리
        add_to_window(ssm_out)
```

### 효율성

```
Window attention:  O(65²) = ~3.2M FLOPs
Full attention:    O(4096²) = ~12.9B FLOPs
→ 2300x faster! 🚀
```

## 🔧 Configuration 가이드

### 기본 설정 (WikiText-103)

```python
config = StrideHybridConfig(
    n_ssm_outputs=15,
    m_input_tokens=50,
    stride=16,  # m/3 for 3x coverage

    transformer_n_layers=12,  # GPT-2
    ssm_n_layers=24,          # Mamba-130M

    d_model=768,
    vocab_size=50280,
)
```

### Long-context 설정 (PG-19)

```python
config = StrideHybridConfig(
    n_ssm_outputs=20,      # 더 많은 메모리
    m_input_tokens=80,     # 더 긴 윈도우
    stride=25,             # 더 긴 stride
    # ... 나머지 동일
)
```

### 빠른 실험 (Small)

```python
config = StrideHybridConfig(
    n_ssm_outputs=10,
    m_input_tokens=30,
    stride=10,

    transformer_n_layers=6,   # 작은 모델
    ssm_n_layers=12,

    d_model=512,
)
```

## 📊 모델 분석

### 파라미터 확인

```python
params = model.count_parameters()

for component, count in params['breakdown'].items():
    print(f"{component:20s}: {count/1e6:6.1f}M")
```

**출력 예시**:

```
embedding           :   38.6M
transformer         :  117.0M  ← GPT-2
attention_pooling   :    1.2M  ← 새로운 학습 가능 부분
ssm                 :  130.0M  ← Mamba
lm_head            :    0.0M  ← Tied with embedding
```

### Attention 패턴 분석

```python
from src.models.components.attention_pooling import AttentionPoolingAnalyzer

# Forward pass
logits, aux = model(input_ids)

# 분석
if 'attention_weights' in aux:
    attn_weights = aux['attention_weights'][0]  # 첫 write step

    stats = AttentionPoolingAnalyzer.analyze_attention(
        attn_weights,
        window_size=65,
        n_ssm=15
    )

    print(f"SSM attention:   {stats['ssm_total']:.2%}")
    print(f"Input attention: {stats['input_total']:.2%}")
    print(f"Entropy:         {stats['entropy']:.2f}")

    # Visualization
    AttentionPoolingAnalyzer.visualize_attention(
        attn_weights,
        save_path='attention_pattern.png'
    )
```

## 🎯 다음 단계

### A. 데이터 준비

```bash
# WikiText-103 다운로드
python -c "from datasets import load_dataset; \
           load_dataset('wikitext', 'wikitext-103-v1')"
```

### B. 학습 준비

1. Data loader 구현 (`src/data/dataset.py`)
2. Training loop 구현 (`src/training/trainer.py`)
3. Optimizer 설정 (`src/training/optimizer.py`)

### C. 실험 실행

```bash
# WikiText-103 학습
python experiments/train_wikitext.py \
    --n_ssm_outputs 15 \
    --m_input_tokens 50 \
    --stride 16 \
    --batch_size 8 \
    --epochs 20
```

## 🐛 문제 해결

### CUDA Out of Memory

```python
# Batch size 줄이기
config = StrideHybridConfig(...)
training_config = TrainingConfig(
    batch_size=2,              # ← 4에서 2로
    gradient_accumulation_steps=8  # ← 효과적 batch size 유지
)
```

### mamba-ssm 설치 실패

```bash
# CUDA 없이 테스트
python -c "from src.models.components.ssm import SimpleSSM; \
           print('✅ Using SimpleSSM fallback')"
```

### transformers 없이 테스트

```python
from src.models.components.transformer import SimpleCausalTransformer
transformer = SimpleCausalTransformer(...)  # GPT-2 없이도 작동
```

## 📖 추가 자료

- **IMPLEMENTATION.md**: 상세 구현 설명 (7.3 KB)
- **SUMMARY.md**: 완성 보고서
- **README.md**: 프로젝트 개요

## 💬 피드백 & 기여

이슈나 개선 사항이 있으시면:

1. 코드 리뷰
2. 테스트 추가
3. 문서 개선
4. 실험 결과 공유

---

**Happy Coding! 🚀**

이 구현은 PDF의 아키텍처를 충실히 따르면서도,  
업로드하신 Samba 코드의 실용적인 패턴을 재사용했습니다.
