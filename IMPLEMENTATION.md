# Stride-based Hybrid SSM-Transformer - 구현 준비 완료

## 📁 프로젝트 구조

```
stride_hybrid_ssm/
├── README.md                          ✅ 프로젝트 소개
├── requirements.txt                   ✅ 의존성
│
├── config/
│   ├── __init__.py                   ✅
│   └── model_config.py               ✅ 모델 및 학습 설정
│
├── src/
│   ├── __init__.py                   ✅
│   ├── models/
│   │   ├── __init__.py               ✅
│   │   ├── stride_hybrid.py          ✅ 메인 모델
│   │   ├── window_manager.py         ✅ 윈도우 관리
│   │   └── components/
│   │       ├── __init__.py           ✅
│   │       ├── attention_pooling.py  ✅ Attention Pooling
│   │       ├── transformer.py        ✅ Windowed Transformer
│   │       └── ssm.py                ✅ SSM (Mamba)
│   │
│   ├── data/                         🔲 TODO
│   ├── training/                     🔲 TODO  
│   └── utils/                        🔲 TODO
│
├── experiments/                       🔲 TODO
└── tests/
    └── test_integration.py           ✅ 통합 테스트
```

## ✅ 구현 완료된 컴포넌트

### 1. 핵심 아키텍처

#### **AttentionPooling** (`attention_pooling.py`)
- Query-Key-Value attention으로 윈도우 압축
- Query: 윈도우 평균의 projection
- Keys: 각 위치의 projection  
- Output: Attention weights로 가중합
- 분석 도구 포함 (attention pattern visualization)

#### **WindowManager** (`window_manager.py`)
- 단일/배치 버전 모두 구현
- SSM outputs (n개) + input tokens (m개) 관리
- Efficient deque 기반 rotation
- 상태 저장/로드 지원

#### **WindowedTransformer** (`transformer.py`)
- GPT-2 기반 (pre-trained weights 로드 가능)
- 작은 윈도우만 처리 (65 tokens)
- SimpleCausalTransformer도 제공 (pretrained 없이 테스트용)

#### **SSMMemory** (`ssm.py`)
- Mamba 기반 (pre-trained weights 로드 가능)
- 24-layer SSM with RMSNorm
- SimpleSSM도 제공 (mamba-ssm 없이 테스트용)

#### **StrideHybridModel** (`stride_hybrid.py`)
- 전체 파이프라인 통합
- Training mode: 전체 시퀀스 처리
- Generation mode: auto-regressive 생성
- 파라미터 카운팅 및 통계 제공

### 2. 설정 시스템

#### **StrideHybridConfig**
- 모든 하이퍼파라미터 정의
- 자동 검증 (stride <= m 등)
- 추천 값 경고
- Pretty print summary

#### **TrainingConfig**
- 학습 관련 설정
- Dataset, optimizer, schedule 등

## 🔧 다음 구현 단계

### Phase 1: Data Pipeline (우선순위 높음)
```python
src/data/
├── dataset.py          # WikiText-103, PG-19 로더
├── tokenizer.py        # GPT-2 tokenizer wrapper
└── data_loader.py      # 효율적인 DataLoader
```

### Phase 2: Training Loop
```python
src/training/
├── trainer.py          # 메인 학습 루프
├── optimizer.py        # AdamW with grouped LR
└── scheduler.py        # Cosine annealing
```

### Phase 3: Utilities
```python
src/utils/
├── metrics.py          # Perplexity 계산
├── checkpoint.py       # 체크포인트 관리
└── logging.py          # Wandb 통합
```

### Phase 4: Experiments
```python
experiments/
├── train_wikitext.py   # WikiText-103 학습
├── train_pg19.py       # PG-19 학습
├── evaluate.py         # 평가 스크립트
└── ablation/           # Ablation studies
```

## 🎯 핵심 설계 결정

### 1. 업로드된 코드와의 차이점

| 특성 | 업로드된 코드 (Samba) | 제안 아키텍처 |
|------|---------------------|-------------|
| **처리 방식** | 24개 Mamba 레이어 전체 통과 | 작은 윈도우만 Transformer 처리 |
| **메모리 사용** | 모든 레이어 출력 저장 | SSM outputs만 저장 (n=15) |
| **업데이트** | 매 스텝 | Stride 간격 (6.25%) |
| **압축 방식** | LSM 선형 믹싱 | Attention Pooling |
| **효율성** | O(L) per layer | O(window²) = O(65²) |

### 2. 재사용한 패턴

#### ✅ Mamba 사용법
```python
# From uploaded code
from mamba_ssm import Mamba as MambaCUDA
from mamba_ssm.ops.triton.layer_norm import RMSNorm

# Residual structure
residual = x
x_norm = layer_norm(x)
x_block = mamba_layer(x_norm)
x = residual + x_block
```

#### ✅ GPT-2 로딩
```python
# From uploaded code  
from transformers import GPT2Model
model = GPT2Model.from_pretrained("gpt2")
output = model(inputs_embeds=x)  # Direct embedding input
```

#### ✅ Weight Loading 패턴
```python
# HuggingFace → Custom model
# 1. Load state dict
# 2. Map keys with prefix replacement
# 3. load_state_dict with strict=False
```

### 3. 독창적인 구현

#### Stride-based Write
```python
if step % stride == 0 and step > 0:
    # Only update 6.25% of steps
    pooled = attention_pooling(window)
    ssm_output = ssm(pooled)
    window_manager.append_ssm(ssm_output)
```

#### Efficient Window Rotation
```python
# deque with maxlen for O(1) rotation
self.ssm_outputs = deque(maxlen=n)
self.input_tokens = deque(maxlen=m)
```

## 📊 성능 예상

### Computational Efficiency

| Component | Complexity | FLOPs (per step) |
|-----------|-----------|------------------|
| Window Attention | O(w²) | ~3.2M |
| Attention Pooling (amortized) | O(w) | ~2.4M |
| SSM (amortized) | O(d²) | ~0.04M |
| **Total** | - | **~5.6M** |
| Full Attention (L=4096) | O(L²) | ~12.9B |
| **Speedup** | - | **~2300x** |

### Memory Usage

| Item | Size |
|------|------|
| Window activations | 50K floats |
| SSM state | 16 (minimal) |
| KV cache (not used) | 0 |
| **vs Full Attention** | **L/(n+m) = 63x less** |

## 🚀 실행 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 테스트 실행
```bash
cd stride_hybrid_ssm
python tests/test_integration.py
```

### 3. 모델 사용 예제
```python
from config.model_config import StrideHybridConfig
from src.models.stride_hybrid import StrideHybridModel

# Create config
config = StrideHybridConfig(
    n_ssm_outputs=15,
    m_input_tokens=50,
    stride=16
)

# Create model
model = StrideHybridModel(config)

# Forward pass
logits, aux = model(input_ids)

# Generate
generated = model.generate(
    input_ids=prompt,
    max_new_tokens=100,
    temperature=0.8
)
```

## 📝 다음 할 일

### 즉시 가능
1. ✅ 테스트 실행하여 기본 동작 확인
2. 🔲 WikiText-103 데이터 로더 구현
3. 🔲 기본 학습 루프 작성
4. 🔲 Pre-trained weights 로딩 테스트

### 이후 단계
5. 🔲 WikiText-103에서 학습
6. 🔲 Perplexity 평가
7. 🔲 Ablation studies (stride, window size)
8. 🔲 Long-context (PG-19) 실험
9. 🔲 Needle-in-haystack 테스트

## 💡 주요 차별점

1. **고정 윈도우 + 동적 압축**: 윈도우 크기는 고정, 압축은 학습 가능
2. **Stride 기반 효율성**: 16 스텝마다 1번만 SSM 업데이트 (6.25%)
3. **명확한 역할 분리**: Transformer=단기, SSM=장기
4. **구현 단순성**: ~50줄 핵심 로직
5. **Pre-trained 활용**: GPT-2 + Mamba 재사용

## 🔍 참고사항

- 모든 코드는 GPU/CPU 양쪽 호환
- mamba-ssm 없이도 SimpleSSM으로 테스트 가능
- transformers 없이도 SimpleCausalTransformer로 테스트 가능
- 통합 테스트로 각 컴포넌트 검증 완료
