# GUIDELINE_SELECTOR — 초록 기반 연구 설계 식별 및 지침 매핑

이 파일은 논문의 초록(Abstract)을 분석하여 연구 설계를 식별하고,
적용할 평가 지침을 자동으로 결정하는 규칙을 담고 있다.

---

## 1. 초록 추출 규칙

### 1-1. 구조화 초록 (Structured Abstract) 처리
- "Abstract", "초록" 헤더 이후 텍스트를 초록으로 인식
- 소제목(Background, Objective, Methods, Results, Conclusion 등)이 있으면 전체를 초록으로 인식
- Methods 소제목 아래 내용을 연구 설계 식별의 **1차 분석 대상**으로 설정

### 1-2. 비구조화 초록 처리
- 제목과 서론(Introduction) 사이의 단락을 초록으로 간주
- 100~350단어 사이의 단락을 초록 후보로 탐색

### 1-3. 초록 없는 경우 대체 전략 (순서대로 시도)
1. 제목(Title) 전체 분석
2. 서론(Introduction) 첫 2개 문단
3. 방법(Methods/Study Design) 섹션 첫 2개 문단

---

## 2. 연구 설계 식별 의사결정 트리

아래 순서대로 조건을 확인하고, 첫 번째 일치하는 분류를 사용한다.

```
[초록 분석 시작]
    │
    ├─ 1순위: "systematic review" / "meta-analysis" / "체계적 문헌고찰" /
    │         "메타분석" / "overview of reviews" 포함?
    │   ├─ YES + ("protocol" / "프로토콜" / "PROSPERO registration") 포함?
    │   │   └─ ★ SR-P (SR 프로토콜)
    │   └─ YES (protocol 없음)
    │       └─ ★ SR (체계적 문헌고찰 / 메타분석)
    │
    ├─ 2순위: "scoping review" / "scoping study" / "주제범위 문헌고찰" /
    │         "evidence map" / "Arksey and O'Malley" 포함?
    │   └─ ★ SR-SCR (Scoping Review)
    │
    ├─ 3순위: "randomized" / "randomised" / "무작위" / "RCT" /
    │         "random allocation" / "random assignment" / "randomly assigned" 포함?
    │   └─ ★ RCT (무작위대조시험)
    │
    ├─ 4순위: ("sensitivity" + "specificity") / "diagnostic accuracy" /
    │         "ROC" / "AUC" / "진단 정확도" / ("민감도" + "특이도") /
    │         "index test" / "reference standard" 포함?
    │   └─ ★ DX (진단 정확도 연구)
    │
    ├─ 5순위: "cohort" / "코호트" / ("prospective" + "follow-up") /
    │         ("retrospective" + ("incidence" / "hazard ratio" / "risk")) 포함?
    │   └─ ★ OBS-COHORT (코호트 연구)
    │
    ├─ 6순위: "case-control" / "case–control" / "환자-대조군" /
    │         ("odds ratio" + ("exposure" / "노출")) / "matched controls" 포함?
    │   └─ ★ OBS-CC (환자-대조군 연구)
    │
    ├─ 7순위: "cross-sectional" / "단면" / ("prevalence" + "survey") /
    │         "one time point" / "at a single point" 포함?
    │   └─ ★ OBS-CS (단면 연구)
    │
    ├─ 8순위: "case report" / "case series" / "case-series" /
    │         "증례보고" / "증례 계열" 포함?
    │   └─ ★ CR (증례보고 / 증례계열)
    │
    └─ 위 조건 모두 불일치 → Methods 섹션으로 확장 분석
        └─ 여전히 불확실 → 사용자에게 확인 요청 (섹션 4 참조)
```

---

## 3. 설계별 상세 키워드 사전

### SR / 메타분석 (SR)
| 유형 | 키워드 |
|------|--------|
| 1차 (직접 명시) | systematic review, meta-analysis, 체계적 문헌고찰, 메타분석, pooled analysis, quantitative synthesis |
| 2차 (방법론 단서) | databases searched, PubMed/Embase/Cochrane, inclusion criteria, PRISMA, heterogeneity, I², forest plot, pooled OR/RR/MD, risk of bias assessment, funnel plot |
| 구조적 단서 | 초록 소제목: "Data Sources", "Study Selection", "Data Extraction", "Data Synthesis" |

### SR 프로토콜 (SR-P)
| 유형 | 키워드 |
|------|--------|
| 1차 | protocol, 프로토콜, review protocol |
| 2차 | PROSPERO, registered, a priori, planned methods, will be searched, eligibility criteria will |
| 구조적 단서 | 결과(Results) 섹션 없이 방법(Methods)만 기술; 미래형 동사 사용 |

### Scoping Review (SR-SCR)
| 유형 | 키워드 |
|------|--------|
| 1차 | scoping review, scoping study, 주제범위 문헌고찰, evidence map, evidence mapping |
| 2차 | charting the data, Arksey and O'Malley, Levac, JBI, breadth of evidence, map the evidence |
| 구조적 단서 | "Results" 대신 "Findings"; 정량적 합성(메타분석) 없음 |

### RCT (RCT)
| 유형 | 키워드 |
|------|--------|
| 1차 | randomized controlled trial, randomised controlled trial, RCT, 무작위대조시험, randomly assigned |
| 2차 | placebo, control group, allocation concealment, blinding, double-blind, intention-to-treat, per-protocol, CONSORT |
| 구조적 단서 | 초록 소제목: "Participants", "Interventions", "Randomization"; 두 군 비교 결과 |

### 코호트 연구 (OBS-COHORT)
| 유형 | 키워드 |
|------|--------|
| 1차 | cohort study, 코호트 연구, prospective study, retrospective cohort, longitudinal study |
| 2차 | follow-up, incidence, hazard ratio, HR, person-years, survival analysis, Kaplan-Meier, exposed/unexposed |
| 구조적 단서 | 시간 경과에 따른 추적 기술; 특정 노출 여부로 그룹 구분 |

### 환자-대조군 연구 (OBS-CC)
| 유형 | 키워드 |
|------|--------|
| 1차 | case-control study, case–control, 환자-대조군, case-referent |
| 2차 | odds ratio, OR (95% CI), matched controls, exposure history, cases and controls |
| 구조적 단서 | 결과(outcome) 여부로 그룹 구분 후 노출 소급 조사 |

### 단면 연구 (OBS-CS)
| 유형 | 키워드 |
|------|--------|
| 1차 | cross-sectional study, cross-sectional survey, 단면연구, prevalence study |
| 2차 | prevalence, at a single point in time, questionnaire survey, one-time measurement |
| 구조적 단서 | 추적관찰(follow-up) 언급 없음; 단일 시점 측정 |

### 진단 정확도 연구 (DX)
| 유형 | 키워드 |
|------|--------|
| 1차 | diagnostic accuracy, diagnostic performance, 진단 정확도, sensitivity and specificity |
| 2차 | ROC curve, AUC, area under the curve, positive predictive value, negative predictive value, likelihood ratio, index test, reference standard, gold standard, cut-off |
| 구조적 단서 | 2×2 분할표 결과 보고; 민감도·특이도·PPV·NPV 수치 |

### 증례보고 / 증례계열 (CR)
| 유형 | 키워드 |
|------|--------|
| 1차 | case report, case series, 증례보고, 증례 계열 |
| 2차 | a rare case of, unusual presentation, first reported, a patient presented |
| 구조적 단서 | 초록 매우 짧음(100단어 이하); 통계 분석 없음; 단일 환자 기술 |

---

## 4. 복합 설계 처리 규칙

| 복합 상황 | 판정 | 적용 지침 |
|-----------|------|----------|
| SR + 메타분석 | SR | PRISMA 2020 + AMSTAR 2 + GRADE |
| SR + RCT만 포함 | SR | PRISMA 2020 + AMSTAR 2 + RoB 2.0(개별연구) + GRADE |
| SR + 관찰연구 포함 | SR | PRISMA 2020 + AMSTAR 2 + ROBINS-I/NOS(개별연구) + GRADE |
| SR + 진단 정확도 포함 | SR | PRISMA 2020 + AMSTAR 2 + QUADAS-2(개별연구) + GRADE |
| Scoping Review + 서술적 합성 | SR-SCR | PRISMA-ScR 2018 |
| RCT + 비무작위 비교 | RCT 우선 | CONSORT + RoB 2.0 |
| 코호트 + 단면 혼합 | 주된 설계 우선 | STROBE + NOS |
| 진단 + 관찰연구 | DX 우선 | STARD + QUADAS-2 |

---

## 5. 설계 → 지침 자동 매핑 표

| 설계 코드 | 보고 지침 파일 | 질/비뚤림 평가 파일 | GRADE |
|-----------|--------------|-------------------|-------|
| SR | guidelines/PRISMA_2020.md | guidelines/AMSTAR2.md + guidelines/ROBIS.md | ✅ |
| SR-P | guidelines/PRISMA-P_2015.md | — | — |
| SR-SCR | guidelines/PRISMA-ScR_2018.md | — | — |
| RCT | guidelines/CONSORT_2010.md | guidelines/RoB2.md | — |
| OBS-COHORT | guidelines/STROBE.md (코호트) | guidelines/NOS.md | — |
| OBS-CC | guidelines/STROBE.md (환자-대조군) | guidelines/NOS.md | — |
| OBS-CS | guidelines/STROBE.md (단면) | guidelines/NOS.md | — |
| DX | guidelines/STARD_2015.md | guidelines/QUADAS-2.md | — |
| CR | guidelines/CARE_2013.md | — | — |

---

## 6. 불확실 시 사용자 확인 프롬프트 템플릿

초록 분석으로 연구 설계를 확정할 수 없을 때, 아래 형식으로 사용자에게 확인을 요청한다:

```
초록 분석 결과, 연구 설계를 명확히 판별하기 어렵습니다.

[초록에서 발견된 단서]
- "..."
- "..."

아래 중 가장 적합한 설계를 선택해 주세요:
1. 체계적 문헌고찰 / 메타분석 (SR)
2. 무작위대조시험 (RCT)
3. 코호트 연구
4. 환자-대조군 연구
5. 단면 연구
6. 진단 정확도 연구
7. 증례보고
8. 기타 (직접 입력)
```

---

## 7. 연구 설계 확정 후 사용자 고지 형식

식별이 완료되면 아래 형식으로 사용자에게 확인을 받고 Phase 2로 진행:

```
📋 연구 설계 식별 결과

확인된 설계: [설계명]
식별 근거:
  - [초록에서 발견된 키워드/표현 인용]
  - [추가 단서]

적용할 평가 지침:
  - 보고 지침: [지침명]
  - 질/비뚤림 평가: [도구명]
  - GRADE 적용: [Yes / No]

이 분류가 맞으면 계속 진행합니다.
다르다면 올바른 설계를 알려주세요.
```
