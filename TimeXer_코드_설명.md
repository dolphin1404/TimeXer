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

#### Features = 'MS' (Multivariate → Single) 모드

```python
# 내생 변수 (예측 타겟): 마지막 변수만 추출
en_data = x_enc[:, :, -1].unsqueeze(-1)  # [B, L, 1]
en_data = en_data.permute(0, 2, 1)       # [B, 1, L]

# 외생 변수: 나머지 변수들
ex_data = x_enc[:, :, :-1]  # [B, L, N-1]
```

**텐서 차원:**
- 내생: `[32, 1, 168]` - 배치 32, 변수 1개, 시퀀스 168
- 외생: `[32, 168, 2]` - 배치 32, 시퀀스 168, 외생 변수 2개

#### Features = 'M' (Multivariate → Multivariate) 모드

```python
# 모든 변수를 내생으로 취급
en_data = x_enc.permute(0, 2, 1)  # [B, N, L]

# 외생도 동일한 데이터 사용
ex_data = x_enc  # [B, L, N]
```

---

### 3단계: 내생 변수 임베딩 (EnEmbedding)

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

```python
# 변수 차원과 배치 차원 병합
x = torch.reshape(x, (B * n_vars, num_patches, patch_len))
# 예: [32, 7, 24]
```

#### 3-3. 임베딩

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

---

### 4단계: 외생 변수 임베딩 (DataEmbedding_inverted)

```python
# 입력: ex_data [B, L, N-1]
# 예: [32, 168, 2]

# 전치 (Variate 축으로 변환)
x = x.permute(0, 2, 1)  # [B, N-1, L]
# 예: [32, 2, 168]

# Linear 임베딩
x = self.value_embedding(x)  # [B, N-1, d_model]
# 예: [32, 2, 512]
```

**Inverted 방식:**
- 시간 축을 임베딩에 사용
- 각 변수가 하나의 토큰이 됨
- 변수 간 관계를 효과적으로 학습

**최종 외생 임베딩:** `[B, N-1, d_model]`
- 예: `[32, 2, 512]`

---

### 5단계: 인코더 처리 (EncoderLayer)

인코더는 여러 레이어로 구성되며, 각 레이어는 3개의 서브레이어를 포함합니다.

#### 5-1. Self-Attention (내생 변수 내부)

```python
# 입력: en_embed [B*n_vars, num_patches+1, d_model]
# 예: [32, 8, 512]

# Self-Attention
x = x + self.dropout(self.self_attention(x, x, x)[0])
# 출력: [32, 8, 512]

x = self.norm1(x)  # Layer Normalization
```

**Self-Attention 동작:**
1. Query, Key, Value 생성: `Q = K = V = x`
2. Attention Score 계산: `Attention(Q,K,V) = softmax(QK^T/√d) × V`
3. 각 패치가 다른 패치들과 정보 교환
4. 글로벌 토큰도 모든 패치와 상호작용

**목적:** 내생 변수의 시간적 의존성 학습

#### 5-2. Cross-Attention (외생→내생)

```python
# 글로벌 토큰만 추출
x_glb_ori = x[:, -1, :].unsqueeze(1)  # [32, 1, 512]

# 배치 복원
B = ex_embed.shape[0]  # 32
x_glb = torch.reshape(x_glb_ori, (B, -1, d_model))  # [32, 1, 512]

# Cross-Attention: Query=글로벌토큰, Key,Value=외생변수
x_glb_attn = self.cross_attention(
    x_glb,      # Query: [32, 1, 512]
    ex_embed,   # Key:   [32, 2, 512]
    ex_embed    # Value: [32, 2, 512]
)[0]
# 출력: [32, 1, 512]

# 재구성 및 잔차 연결
x_glb_attn = torch.reshape(x_glb_attn, (B*n_vars, 1, d_model))  # [32, 1, 512]
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

```python
# 패치들과 업데이트된 글로벌 토큰 결합
x = torch.cat([x[:, :-1, :], x_glb], dim=1)  # [32, 8, 512]

# Feed-Forward Network
y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
# [32, d_ff, 8] → [32, 2048, 8]

y = self.dropout(self.conv2(y).transpose(-1, 1))
# [32, 2048, 8] → [32, 512, 8] → [32, 8, 512]

# 잔차 연결 및 정규화
output = self.norm3(x + y)  # [32, 8, 512]
```

**FFN (Feed-Forward Network):**
- 2개의 Linear 변환 (확장 → 축소)
- 각 토큰에 독립적으로 적용
- 비선형성 추가 (ReLU/GELU)

#### 인코더 레이어 반복

```python
# 여러 레이어 반복 (e_layers=2 or 3)
for layer in self.layers:
    x = layer(x, cross=ex_embed)
```

---

### 6단계: 예측 헤드 (FlattenHead)

```python
# 입력: enc_out [B*n_vars, num_patches+1, d_model]
# 예: [32, 8, 512]

# 재구성
enc_out = torch.reshape(enc_out, (B, n_vars, num_patches+1, d_model))
# 예: [32, 1, 8, 512]

# 전치
enc_out = enc_out.permute(0, 1, 3, 2)
# 예: [32, 1, 512, 8]

# Flatten: d_model × num_patches 차원으로 펼침
x = self.flatten(enc_out)  # [32, 1, 512*8] = [32, 1, 4096]

# Linear: 예측 길이로 투영
x = self.linear(x)  # [32, 1, pred_len]
# 예: [32, 1, 24]

# 전치
dec_out = x.permute(0, 2, 1)  # [32, 24, 1]
```

**예측 헤드 동작:**
1. 모든 패치와 글로벌 토큰 정보를 하나의 벡터로 병합
2. Linear 변환으로 예측 길이만큼 출력
3. 각 시점의 예측값 생성

---

### 7단계: 역정규화 (De-Normalization)

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

---

## 실행 순서

### 전체 Forward Pass 순서

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
