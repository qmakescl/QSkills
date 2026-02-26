# ROBINS-I — 비무작위 중재 연구 비뚤림 위험 평가 도구

**정식 명칭:** Risk Of Bias In Non-randomised Studies of Interventions
**발표 연도:** 2016
**구성:** 7개 도메인 + 신호 질문
**주요 참고 문헌:** Sterne JAC et al. BMJ 2016;355:i4919
**공식 사이트:** https://methods.cochrane.org/robins-i

---

## 개요

ROBINS-I는 **비무작위 중재 연구(NRSI: Non-Randomised Studies of Interventions)**의
비뚤림 위험을 평가하기 위한 도구입니다.

각 비무작위 연구를 '가상의 무작위 시험(target trial)'을 모방하는 것으로 개념화하여,
무작위화가 이루어지지 않은 결과로 나타나는 편향을 체계적으로 파악합니다.

> 적용 대상: 코호트, 환자-대조군 등 비무작위 중재 연구 (노출 연구는 ROBINS-E 사용)

---

## 7개 비뚤림 도메인

### [기저선 이전 도메인 - 중재 시작 전]

#### 도메인 1: 교란으로 인한 비뚤림 (Bias due to confounding)

**우려 내용:** 혼란변수(confounder)가 결과에 미치는 영향 미통제

신호 질문:
- 1.1 교란변수가 선험적으로 식별되었는가?
- 1.2 모든 중요한 교란변수를 측정하였는가?
- 1.3 교란변수에 대한 균형(balancing)이 이루어졌는가?
- 1.4 잔류 교란(residual confounding)이 있을 가능성이 있는가?

---

#### 도메인 2: 연구 참여자 선택 비뚤림 (Bias in selection of participants into the study)

**우려 내용:** 선택 기준이 결과 또는 중재와 연관되어 있을 경우

신호 질문:
- 2.1 참여자 선택이 결과나 결과의 예측 변수와 연관되어 있는가?
- 2.2 중재 시작 시점이 아닌 다른 시점에서 참여자를 선택하였는가?

---

#### 도메인 3: 중재 분류 비뚤림 (Bias in classification of interventions)

**우려 내용:** 중재 노출의 측정 오류

신호 질문:
- 3.1 중재 상태 정의가 명확하고 측정 가능한가?
- 3.2 중재 상태 측정에 정보 편향이 존재하는가?

---

### [기저선 이후 도메인 - 중재 시작 후]

#### 도메인 4: 의도 중재에서의 이탈로 인한 비뚤림
(Bias due to departures from intended interventions)

신호 질문:
- 4.1 코-중재(co-intervention)가 적용되었는가?
- 4.2 참여자가 중재를 변경하거나 중단하였는가?
- 4.3 분석이 배정된 중재를 기준으로 적절히 수행되었는가?

---

#### 도메인 5: 결측 데이터로 인한 비뚤림 (Bias due to missing data)

신호 질문:
- 5.1 중재 또는 결과 데이터가 누락되었는가?
- 5.2 누락의 증거 또는 근거가 있는가?
- 5.3 적절한 분석 방법으로 누락 데이터를 처리하였는가?

---

#### 도메인 6: 결과 측정 비뚤림 (Bias in measurement of outcomes)

신호 질문:
- 6.1 결과 측정이 중재 상태에 대한 지식에 영향을 받았는가?
- 6.2 결과 정의 및 측정 방법이 그룹 간 동일한가?
- 6.3 결과 측정 도구가 타당화되었는가?

---

#### 도메인 7: 보고 결과 선택 비뚤림 (Bias in selection of the reported result)

신호 질문:
- 7.1 분석 계획이 데이터 수집 전에 등록·공개되었는가?
- 7.2 여러 가능한 결과 측정 방법 중 선택적으로 보고되었는가?
- 7.3 특정 분석 결과만 선택적으로 보고되었는가?

---

## 비뚤림 판정 등급

각 도메인 및 전반적 판정:

| 판정 | 의미 |
|------|------|
| **Low** | 낮은 비뚤림 위험 |
| **Moderate** | 중간 비뚤림 위험 (해석 시 주의) |
| **Serious** | 심각한 비뚤림 위험 |
| **Critical** | 매우 심각한 비뚤림 위험 (결과 신뢰 불가) |
| **No information** | 판단 불가 |

---

## 전반적 판정 규칙

- 도메인 중 하나라도 '심각(Serious)' → 전반적 '심각'
- 도메인 중 하나라도 '치명적(Critical)' → 전반적 '치명적'
- 모든 도메인이 '낮음(Low)' → 전반적 '낮음'

---

## ROBINS-I vs. Cochrane RoB 2 비교

| 구분 | ROBINS-I | RoB 2 |
|------|----------|-------|
| 적용 연구 | 비무작위 중재 연구 | 무작위대조시험(RCT) |
| 도메인 수 | 7개 | 5개 |
| 판정 등급 | Low/Moderate/Serious/Critical | Low/Some concerns/High |
| 교란 처리 | 별도 도메인 포함 | 무작위화로 통제 가정 |

---

## 참고

- 공식 출판: BMJ 2016;355:i4919
- 안내 문서: https://www.bristol.ac.uk/media-library/sites/social-community-medicine/images/centres/cresyda/ROBINS-I_detailed_guidance.pdf
- ROBINS-E: 비무작위 노출 연구(exposure study)에 사용하는 도구
