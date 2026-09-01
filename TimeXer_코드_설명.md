# TimeXer 코드 동작 원리 설명

## 목차
1. [개요](#개요)
2. [전체 아키텍처](#전체-아키텍처)
3. [데이터 흐름과 텐서 차원 변화](#데이터-흐름과-텐서-차원-변화)
4. [주요 컴포넌트 상세 설명](#주요-컴포넌트-상세-설명)
5. [실행 순서](#실행-순서)

---

## 개요

TimeXer는 **외생 변수(exogenous variables)**를 활용한 시계열 예측을 위한 Transformer 기반 모델입니다. 

### 핵심 개념
- **내생 변수(Endogenous)**: 예측하려는 목표 시계열 데이터
- **외생 변수(Exogenous)**: 예측에 도움이 되는 추가 변수들 (예: 날씨, 경제 지표 등)
- **패치 기반 표현**: 내생 변수를 작은 패치로 나누어 처리
- **변수 레벨 표현**: 외생 변수를 변수 단위로 처리
- **글로벌 토큰**: 내생과 외생 정보를 연결하는 다리 역할

---

## 전체 아키텍처

```
입력 데이터 (x_enc)
    ↓
┌─────────────────────────────────────┐
│  1. 정규화 (Normalization)          │
│     - 평균 제거, 표준편차로 나눔      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. 데이터 분리                      │
│     - 내생 변수: x[:,:,-1]          │
│     - 외생 변수: x[:,:,:-1]         │
└─────────────────────────────────────┘
    ↓                    ↓
    ↓              ┌──────────────────┐
    ↓              │ 외생 임베딩       │
    ↓              │ (변수 레벨)      │
    ↓              └──────────────────┘
    ↓                    ↓
┌────────────┐           ↓
│ 내생 임베딩 │           ↓
│ (패치 레벨)│           ↓
│ + 글로벌   │           ↓
│   토큰     │           ↓
└────────────┘           ↓
    ↓                    ↓
    └────────┬───────────┘
             ↓
    ┌─────────────────┐
    │  3. 인코더       │
    │  - Self-Attn    │
    │  - Cross-Attn   │
    │  - FFN          │
    └─────────────────┘
             ↓
    ┌─────────────────┐
    │  4. 예측 헤드    │
    │  - Flatten      │
    │  - Linear       │
    └─────────────────┘
             ↓
    ┌─────────────────┐
    │  5. 역정규화     │
    └─────────────────┘
             ↓
         예측 결과
```

---

## 데이터 흐름과 텐서 차원 변화

### 입력 단계

**초기 입력:**
- `x_enc`: `[B, L, N]` 
  - B: 배치 크기 (Batch size)
  - L: 시퀀스 길이 (Sequence length, 예: 168)
  - N: 전체 변수 개수 (내생 1개 + 외생 변수들)

**예시:** `[32, 168, 3]` - 배치 32, 시퀀스 길이 168, 변수 3개

---

### 1단계: 정규화 (Normalization)

**📁 위치:** `models/TimeXer.py` - `forecast()` 메서드 내

```python
# 평균 계산
means = x_enc.mean(1, keepdim=True)  # [B, 1, N]

# 평균 제거
x_enc = x_enc - means  # [B, L, N]

# 표준편차 계산
stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, N]

# 정규화
x_enc = x_enc / stdev  # [B, L, N]
```

**목적:** 시계열 데이터의 스케일을 통일하여 학습 안정성 향상

---

### 2단계: 내생/외생 변수 분리

**📁 위치:** `models/TimeXer.py` - `forecast()` 메서드 내

#### Features = 'MS' (Multivariate → Single) 모드

```python
# 실제 코드 (MS 모드, forecast)
en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
```

**텐서 차원:**
- 내생: `[32, 1, 168]` - 배치 32, 변수 1개, 시퀀스 168
- 외생: `[32, 168, 2]` - 배치 32, 시퀀스 168, 외생 변수 2개

#### Features = 'M' (Multivariate → Multivariate) 모드

**📁 위치:** `models/TimeXer.py` - `forecast_multi()` 메서드 내

```python
# 실제 코드 (M 모드, forecast_multi)
en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
ex_embed = self.ex_embedding(x_enc, x_mark_enc)
```

---

### 3단계: 내생 변수 임베딩 (EnEmbedding)

**📁 위치:** `models/TimeXer.py` - `EnEmbedding` 클래스

#### 3-1. 패치 생성 (Patching)

```python
# 입력: en_data [B, n_vars, L]
# 예: [32, 1, 168]

# unfold로 패치 생성
patch_len = 24  # 패치 길이
x = x.unfold(dimension=-1, size=patch_len, step=patch_len)
# 결과: [B, n_vars, num_patches, patch_len]
# 예: [32, 1, 7, 24]  (168 / 24 = 7개 패치)
```

**패치란?**
- 긴 시계열을 작은 조각(패치)으로 나눔
- 168 시점 → 24시점씩 7개 패치
- 각 패치는 하나의 토큰처럼 처리됨

#### 3-2. 배치 재구성

**📁 위치:** `models/TimeXer.py` - `EnEmbedding.forward()`

```python
# 변수 차원과 배치 차원 병합
x = torch.reshape(x, (B * n_vars, num_patches, patch_len))
# 예: [32, 7, 24]
```

#### 3-3. 임베딩

**📁 위치:** `models/TimeXer.py` - `EnEmbedding.forward()`

```python
# Linear 변환으로 임베딩
x = self.value_embedding(x)  # [B*n_vars, num_patches, d_model]
# 예: [32, 7, 512]

# 위치 임베딩 추가
x = x + self.position_embedding(x)  # [32, 7, 512]
```

**위치 임베딩:**
- 각 패치의 시간적 위치 정보 제공
- sin/cos 함수 기반 인코딩

#### 3-4. 글로벌 토큰 추가

**📁 위치:** `models/TimeXer.py` - `EnEmbedding.forward()`

```python
# 원래 형태로 재구성
x = torch.reshape(x, (B, n_vars, num_patches, d_model))
# 예: [32, 1, 7, 512]

# 글로벌 토큰 생성 및 추가
glb_token = self.glb_token.repeat(B, 1, 1, 1)  # [B, n_vars, 1, d_model]
x = torch.cat([x, glb_token], dim=2)  # [B, n_vars, num_patches+1, d_model]
# 예: [32, 1, 8, 512]  (7개 패치 + 1개 글로벌 토큰)

# 최종 재구성
x = torch.reshape(x, (B * n_vars, num_patches+1, d_model))
# 예: [32, 8, 512]
```

**글로벌 토큰의 역할:**
- 전체 내생 변수 정보를 요약
- 외생 변수와의 연결 다리
- Cross-Attention의 Query로 사용됨

**최종 내생 임베딩:** `[B*n_vars, num_patches+1, d_model]`
- 예: `[32, 8, 512]`

실제 구현 코드 (원문 발췌): `models/TimeXer.py` - `EnEmbedding`

```python
class EnEmbedding(nn.Module):
   def __init__(self, n_vars, d_model, patch_len, dropout):
      super(EnEmbedding, self).__init__()
      # Patching
      self.patch_len = patch_len

      self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
      self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
      self.position_embedding = PositionalEmbedding(d_model)

      self.dropout = nn.Dropout(dropout)

   def forward(self, x):
      # do patching
      n_vars = x.shape[1]
      glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

      x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
      x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
      # Input encoding
      x = self.value_embedding(x) + self.position_embedding(x)
      x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
      x = torch.cat([x, glb], dim=2)
      x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
      return self.dropout(x), n_vars
```

---

### 4단계: 외생 변수 임베딩 (DataEmbedding_inverted)

**📁 위치:** `layers/Embed.py` - `DataEmbedding_inverted` 클래스 (외생 변수 임베딩)

```python
# 입력: ex_data [B, L, N-1]
# 예: [32, 168, 2]

# 전치 (Variate 축으로 변환)
x = x.permute(0, 2, 1)  # [B, N-1, L]
# 예: [32, 2, 168]

# Linear 임베딩
# 주의: 구현상 x_mark_enc가 있으면 [x, x_mark.permute(0,2,1)]를 채널축으로 concat한 뒤 선형 투영합니다.
x = self.value_embedding(x)  # [B, N-1, d_model]
# 예: [32, 2, 512]

실제 구현 코드 (원문 발췌): `layers/Embed.py` - `DataEmbedding_inverted`

```python
class DataEmbedding_inverted(nn.Module):
   def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
      super(DataEmbedding_inverted, self).__init__()
      self.value_embedding = nn.Linear(c_in, d_model)
      self.dropout = nn.Dropout(p=dropout)

   def forward(self, x, x_mark):
      x = x.permute(0, 2, 1)
      # x: [Batch Variate Time]
      if x_mark is None:
         x = self.value_embedding(x)
      else:
         x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
      # x: [Batch Variate d_model]
      return self.dropout(x)
```
```

**Inverted 방식:**
- 시간 축을 임베딩에 사용
- 각 변수가 하나의 토큰이 됨
- 변수 간 관계를 효과적으로 학습

**최종 외생 임베딩:** `[B, N-1, d_model]`
- 예: `[32, 2, 512]`

---

### 5단계: 인코더 처리 (EncoderLayer)

**📁 위치:** `models/TimeXer.py` - `Encoder`, `EncoderLayer` 클래스

인코더는 여러 레이어로 구성되며, 각 레이어는 3개의 서브레이어를 포함합니다.

#### 5-1. Self-Attention (내생 변수 내부)

**📁 위치:** `models/TimeXer.py` - `EncoderLayer.forward()`

```python
# 입력: en_embed [B*n_vars, num_patches+1, d_model]
# 예: [32, 8, 512]

# Self-Attention (실제 구현)
x = x + self.dropout(self.self_attention(
   x, x, x,
   attn_mask=x_mask,
   tau=tau, delta=None
)[0])
x = self.norm1(x)  # Layer Normalization
```

**Self-Attention 동작:**
1. Query, Key, Value 생성: `Q = K = V = x`
2. Attention Score 계산: `Attention(Q,K,V) = softmax(QK^T/√d) × V`
3. 각 패치가 다른 패치들과 정보 교환
4. 글로벌 토큰도 모든 패치와 상호작용

**목적:** 내생 변수의 시간적 의존성 학습

#### 5-2. Cross-Attention (외생→내생)

**📁 위치:** `models/TimeXer.py` - `EncoderLayer.forward()`

```python
# 글로벌 토큰만 추출 (실제 구현)
x_glb_ori = x[:, -1, :].unsqueeze(1)
x_glb = torch.reshape(x_glb_ori, (B, -1, D))
x_glb_attn = self.dropout(self.cross_attention(
   x_glb, cross, cross,
   attn_mask=cross_mask,
   tau=tau, delta=delta
)[0])
x_glb_attn = torch.reshape(x_glb_attn,
                     (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
x_glb = x_glb_ori + x_glb_attn
x_glb = self.norm2(x_glb)
```

**Cross-Attention 동작:**
1. Query: 내생의 글로벌 토큰
2. Key, Value: 외생 변수들
3. 외생 정보를 글로벌 토큰에 통합
4. Attention Score = softmax(Q × K^T / √d)
5. 출력 = Attention Score × V

**목적:** 외생 변수 정보를 내생 변수에 주입

#### 5-3. 재결합 및 FFN

**📁 위치:** `models/TimeXer.py` - `EncoderLayer.forward()`

```python
# 패치들과 업데이트된 글로벌 토큰 결합
y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

# Feed-Forward Network (1x1 Conv로 구현된 두 단계 FFN)
y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
y = self.dropout(self.conv2(y).transpose(-1, 1))

# 잔차 연결 및 정규화
output = self.norm3(x + y)
```

**FFN (Feed-Forward Network):**
- 2개의 Linear 변환 (확장 → 축소)
- 각 토큰에 독립적으로 적용
- 비선형성 추가 (ReLU/GELU)

#### 인코더 레이어 반복

**📁 위치:** `models/TimeXer.py` - `Encoder.forward()`

```python
# 여러 레이어 반복 (e_layers=2 or 3)
for layer in self.layers:
    x = layer(x, cross=ex_embed)
```

---

### 6단계: 예측 헤드 (FlattenHead)

**📁 위치:** `models/TimeXer.py` - `FlattenHead` 클래스

```python
# FlattenHead 실제 구현
class FlattenHead(nn.Module):
   def __init__(self, n_vars, nf, target_window, head_dropout=0):
      super().__init__()
      self.n_vars = n_vars
      self.flatten = nn.Flatten(start_dim=-2)
      self.linear = nn.Linear(nf, target_window)
      self.dropout = nn.Dropout(head_dropout)

   def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
      x = self.flatten(x)
      x = self.linear(x)
      x = self.dropout(x)
      return x
```

**예측 헤드 동작:**
1. 모든 패치와 글로벌 토큰 정보를 하나의 벡터로 병합
2. Linear 변환으로 예측 길이만큼 출력
3. 각 시점의 예측값 생성

---

### 7단계: 역정규화 (De-Normalization)

**📁 위치:** `models/TimeXer.py` - `forecast()` 메서드 내

```python
# 저장된 표준편차와 평균으로 복원
dec_out = dec_out * stdev[:, 0, -1:].unsqueeze(1).repeat(1, pred_len, 1)
dec_out = dec_out + means[:, 0, -1:].unsqueeze(1).repeat(1, pred_len, 1)
```

**최종 출력:** `[B, pred_len, 1]`
- 예: `[32, 24, 1]` - 배치 32, 예측 24시점, 변수 1개

---

## 주요 컴포넌트 상세 설명

### 1. Patching (패치화)

**📁 위치:** `models/TimeXer.py` - `EnEmbedding` 클래스

**원리:**
```
원본 시계열: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
패치 크기 4로 나누기
↓
패치 1: [1, 2, 3, 4]
패치 2: [5, 6, 7, 8]
패치 3: [9, 10, 11, 12]
```

**장점:**
- 지역적 시간 패턴 포착
- 계산 효율성 향상
- Vision Transformer의 패치 개념 차용

### 2. Global Token (글로벌 토큰)

**📁 위치:** `models/TimeXer.py` - `EnEmbedding.__init__()` 및 `forward()`

**역할:**
- 전체 시계열 정보의 요약
- 외생 변수와의 통신 채널
- 학습 가능한 파라미터

**동작:**
```
패치들: [P1, P2, P3, ..., P7]
글로벌: [G]
결합:   [P1, P2, P3, ..., P7, G]

Self-Attention 후:
- P1, P2, ..., P7은 서로 정보 교환
- G는 모든 패치와 정보 교환
- G가 전체 정보를 집약

Cross-Attention:
- G를 Query로 사용
- 외생 변수들을 Key, Value로 사용
- 외생 정보를 G에 통합
```

### 3. Attention Mechanism (어텐션 메커니즘)

**📁 위치:** `layers/SelfAttention_Family.py` - `FullAttention`, `AttentionLayer` 클래스

**Self-Attention 수식:**
```
Q = x × W_Q  (Query 생성)
K = x × W_K  (Key 생성)
V = x × W_V  (Value 생성)

Attention(Q,K,V) = softmax(QK^T / √d_k) × V
```

**Cross-Attention 수식:**
```
Q = x_glb × W_Q      (내생 글로벌 토큰)
K = x_ex × W_K       (외생 변수)
V = x_ex × W_V       (외생 변수)

Attention(Q,K,V) = softmax(QK^T / √d_k) × V
```

**Multi-Head Attention:**
- 여러 개의 어텐션 헤드 병렬 실행
- 다양한 관점에서 관계 학습
- 헤드 출력을 연결(concat)하여 최종 출력

실제 구현 코드 (원문 발췌): `layers/SelfAttention_Family.py` - `AttentionLayer`

```python
class AttentionLayer(nn.Module):
   def __init__(self, attention, d_model, n_heads, d_keys=None,
             d_values=None):
      super(AttentionLayer, self).__init__()

      d_keys = d_keys or (d_model // n_heads)
      d_values = d_values or (d_model // n_heads)

      self.inner_attention = attention
      self.query_projection = nn.Linear(d_model, d_keys * n_heads)
      self.key_projection = nn.Linear(d_model, d_keys * n_heads)
      self.value_projection = nn.Linear(d_model, d_values * n_heads)
      self.out_projection = nn.Linear(d_values * n_heads, d_model)
      self.n_heads = n_heads

   def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
      B, L, _ = queries.shape
      _, S, _ = keys.shape
      H = self.n_heads

      queries = self.query_projection(queries).view(B, L, H, -1)
      keys = self.key_projection(keys).view(B, S, H, -1)
      values = self.value_projection(values).view(B, S, H, -1)

      out, attn = self.inner_attention(
         queries,
         keys,
         values,
         attn_mask,
         tau=tau,
         delta=delta
      )
      out = out.view(B, L, -1)

      return self.out_projection(out), attn
```

또한, 인코더는 다음과 같이 구성됩니다 (원문 발췌): `models/TimeXer.py` - `Model.__init__`

```python
self.encoder = Encoder(
   [
      EncoderLayer(
         AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                       output_attention=False),
            configs.d_model, configs.n_heads),
         AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                       output_attention=False),
            configs.d_model, configs.n_heads),
         configs.d_model,
         configs.d_ff,
         dropout=configs.dropout,
         activation=configs.activation,
      )
      for l in range(configs.e_layers)
   ],
   norm_layer=torch.nn.LayerNorm(configs.d_model)
)
```

---

## 실행 순서

### 전체 Forward Pass 순서

**📁 위치:** `models/TimeXer.py` - `forecast()` 메서드

```
1. 입력 데이터 준비
   └─> x_enc: [B, L, N]

2. 정규화 (use_norm=True인 경우)
   └─> 평균 제거, 표준편차 나눔

3. 데이터 분리
   ├─> 내생: x_enc[:,:,-1] 또는 x_enc (features 모드에 따라)
   └─> 외생: x_enc[:,:,:-1] 또는 x_enc

4. 내생 임베딩
   ├─> 패치 생성: unfold 연산
   ├─> Linear 임베딩 + 위치 임베딩
   └─> 글로벌 토큰 추가

5. 외생 임베딩
   ├─> 차원 전치 (시간→변수 중심)
   └─> Linear 임베딩

6. 인코더 처리 (여러 레이어 반복)
   각 레이어마다:
   ├─> Self-Attention (내생 내부)
   ├─> Cross-Attention (외생→내생)
   └─> Feed-Forward Network

7. 예측 헤드
   ├─> Flatten (모든 정보 병합)
   └─> Linear (예측 길이로 투영)

8. 역정규화
   └─> 원래 스케일로 복원

9. 출력
   └─> 예측 결과: [B, pred_len, 1 or N]
```

### 차원 변화 요약표

| 단계 | 내생 변수 | 외생 변수 | 설명 |
|------|-----------|-----------|------|
| 입력 | [B, L, N] | [B, L, N] | 원본 데이터 |
| 분리 | [B, 1, L] | [B, L, N-1] | MS 모드 기준 |
| 패치 | [B, 1, 7, 24] | - | unfold 연산 |
| 임베딩 | [B, 1, 8, 512] | [B, 2, 512] | +글로벌 토큰 |
| 인코더 입력 | [32, 8, 512] | [32, 2, 512] | 배치 병합 |
| 인코더 출력 | [32, 8, 512] | - | 동일 차원 |
| 헤드 입력 | [32, 1, 512, 8] | - | 재구성 |
| 헤드 출력 | [32, 1, 24] | - | 예측 |
| 최종 출력 | [32, 24, 1] | - | 전치 |

*예시 값: B=32, L=168, N=3, patch_len=24, d_model=512, pred_len=24*

---

## 핵심 설계 아이디어

### 1. 이중 표현 전략
- **내생 변수**: 패치 레벨 표현 → 시간적 의존성 포착
- **외생 변수**: 변수 레벨 표현 → 변수 간 관계 포착

### 2. 글로벌 토큰을 통한 정보 통합
- Self-Attention으로 내생 정보 집약
- Cross-Attention으로 외생 정보 흡수
- 양방향 정보 흐름 실현

### 3. 계층적 처리
```
지역 패턴 (Self-Attention)
    ↓
글로벌 집약 (Global Token)
    ↓
외부 정보 통합 (Cross-Attention)
    ↓
최종 예측 (FFN + Head)
```

### 4. 효율적인 어텐션
- 내생: 패치 수만큼만 어텐션 (7~8개)
- 외생: 변수 수만큼만 어텐션 (2~10개)
- 원본 시퀀스 길이(168)보다 훨씬 효율적

---

## 예제: 구체적인 수치로 이해하기

**시나리오:**
- 전력 가격 예측
- 168시간 과거 데이터로 24시간 예측
- 내생: 전력 가격 (1개)
- 외생: 수요량, 재생에너지 발전량 (2개)

**데이터 흐름:**

1. **입력**: `[32, 168, 3]`
   - 32개 샘플, 168시간, 3개 변수

2. **정규화 후**: `[32, 168, 3]` (값만 변경)

3. **분리**:
   - 내생: `[32, 1, 168]` (전력 가격)
   - 외생: `[32, 168, 2]` (수요량, 재생에너지)

4. **내생 패치화** (patch_len=24):
   - `[32, 1, 7, 24]` (7개 패치)
   - 각 패치 = 24시간 데이터

5. **내생 임베딩**:
   - `[32, 1, 7, 512]` (임베딩)
   - `[32, 1, 8, 512]` (글로벌 토큰 추가)
   - `[32, 8, 512]` (재구성)

6. **외생 임베딩**:
   - `[32, 2, 512]` (2개 변수)

7. **인코더**:
   - Self-Attention: 7개 패치 + 1개 글로벌이 상호작용
   - Cross-Attention: 글로벌이 2개 외생 변수와 상호작용
   - 글로벌 토큰에 모든 정보 집약

8. **예측 헤드**:
   - `[32, 1, 512×8]` → `[32, 1, 4096]` (flatten)
   - `[32, 1, 24]` (linear)
   - `[32, 24, 1]` (전치)

9. **출력**: `[32, 24, 1]`
   - 다음 24시간 전력 가격 예측

---

## 코드 주요 부분 설명

### Forward 함수 (forecast 메서드)

**📁 위치:** `models/TimeXer.py` - `forecast()` 메서드

```python
def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # 1. 정규화
    if self.use_norm:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
    
    # 2. 임베딩
    en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
    ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)
    
    # 3. 인코더
    enc_out = self.encoder(en_embed, ex_embed)
    
    # 4. 재구성 및 예측 헤드
    enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
    enc_out = enc_out.permute(0, 1, 3, 2)
    dec_out = self.head(enc_out)
    dec_out = dec_out.permute(0, 2, 1)
    
    # 5. 역정규화
    if self.use_norm:
        dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
    
    return dec_out
```

---

## 요약

TimeXer는 다음과 같은 독창적인 방법으로 외생 변수를 활용합니다:

1. **이중 임베딩 전략**
   - 내생: 패치 기반 (시간적 패턴)
   - 외생: 변수 기반 (변수 관계)

2. **글로벌 토큰**
   - 내생 정보 집약
   - 외생 정보 수신
   - 정보 통합의 핵심

3. **효율적인 어텐션**
   - Self: 내생 내부 의존성
   - Cross: 외생→내생 전달
   - 계산량 최소화

4. **단순한 Transformer 구조**
   - 복잡한 디코더 없음
   - 인코더만으로 예측
   - 효율적이고 효과적

이 구조로 TimeXer는 외생 변수를 효과적으로 활용하여 정확한 시계열 예측을 수행합니다.

---

## 부록: 텐서 모양 시각화 스크립트 사용법

문서의 텐서 흐름을 PPT에 바로 넣을 수 있도록, 텐서 모양을 단계별로 출력하고 간단한 다이어그램(PNG)으로 저장하는 스크립트를 추가했습니다.

- 스크립트 경로: `scripts/shape_flow_demo.py`
- 주요 인자: `--B --L --N --patch_len --d_model --pred_len --features`
- 출력: 콘솔에 단계별 모양 목록, PNG 파일(`figures/shape_flow.png` 기본값)

Windows PowerShell에서 실행 예시:

```powershell
python .\scripts\shape_flow_demo.py --B 32 --L 168 --N 3 --patch_len 24 --d_model 512 --pred_len 24 --features MS --png figures\shape_flow_MS.png
python .\scripts\shape_flow_demo.py --B 16 --L 96  --N 7 --patch_len 24 --d_model 256 --pred_len 24 --features M  --png figures\shape_flow_M.png
```

참고 사항:
- MS 모드: 내생 1개, 외생 N-1개로 분리되어 내생은 패치+글로벌, 외생은 변수 토큰으로 임베딩됩니다.
- M 모드: 모든 변수가 내생으로 처리되어 `EnEmbedding` 입력의 `n_vars=N`가 됩니다. 외생 임베딩 또한 전체를 사용합니다.
- `DataEmbedding_inverted`는 구현상 `x_mark_enc`가 제공되면 시간 인코딩을 `permute` 후 채널 방향으로 concat하여 선형 임베딩합니다.
