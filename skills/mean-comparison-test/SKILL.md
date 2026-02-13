---
name: mean-comparison-test
description: >
  통계적 가설검정에서 평균비교 분석을 수행하는 스킬. 
  독립표본 t-test(Two-sample/Welch's), 대응표본 t-test(Paired), 
  일원분산분석(One-way ANOVA/Welch's) 및 사후검정(Scheffé, Tukey, Games-Howell 등)을 포함한다.
  사용자가 "평균 비교", "t-test", "t검정", "분산분석", "ANOVA", "집단 간 차이", 
  "그룹별 평균", "사후검정", "post-hoc", "두 집단 비교", "평균차이 검정" 등을 
  언급하면 이 스킬을 사용한다. 데이터 파일(CSV/Excel)을 업로드하여 분석하며, 
  정규성/등분산 검정, 비모수 검정 필요성 안내, 효과크기, 정규분포 곡선 차트, 결과 보고서를 자동 생성한다.
author: yoonani(from CultureLab, aka Q)
version: 2026-02-13
source: https://github.com/qmakescl/QSkills
---

# 평균비교 검정 (Mean Comparison Test) Skill

## 개요

이 스킬은 사용자가 업로드한 데이터(CSV/Excel)를 기반으로 집단 간 평균 비교 가설검정을 수행한다.
분석 유형을 판별하고, 정규성 및 등분산성을 검정한 뒤 적절한 통계분석을 실시하며, 차트와 보고서를 자동 생성한다.

## 워크플로우

### Step 0: 데이터 로드 및 탐색

1. 사용자가 업로드한 CSV 또는 Excel 파일을 pandas로 읽는다.
2. 데이터의 변수명, 타입, 결측치 현황을 간략히 보여준다.
3. 사용자에게 **종속변수**(연속형)와 **독립변수**(집단 구분 변수)를 확인받는다.

### Step 1: 검정 유형 판별

사용자의 가설과 데이터 구조를 파악하여 다음 중 하나를 결정한다:

| 조건 | 검정 유형 |
|------|-----------|
| 독립된 두 집단 비교 | Independent two-sample t-test 또는 Welch's t-test |
| 동일 대상 사전-사후 비교 | Paired t-test |
| 세 집단 이상 비교 | One-way ANOVA |

독립변수의 고유값 수로 자동 판별하되, 대응표본 여부는 사용자에게 확인한다.

### Step 2: 가설 방향 결정 (t-test인 경우)

사용자가 제시한 가설 문장을 분석하여 검정 방향을 결정한다:

- **"차이가 있다", "다르다"** → **양쪽검정 (two-sided)**
- **"보다 크다", "높다", "많다"** → **한쪽검정 (greater)**  
- **"보다 작다", "낮다", "적다"** → **한쪽검정 (less)**

방향이 모호하면 사용자에게 확인한다.

### Step 2.5: 정규성 검정 (Shapiro-Wilk)

- 각 집단의 데이터가 정규분포를 따르는지 검정한다.
- **경고**: n ≤ 20 이거나 정규성 가정이 기각(p < 0.05)되면, 결과 보고서에 **비모수 검정 권장 문구**를 포함한다.
  - 독립 2집단 → Mann-Whitney U test
  - 대응표본 → Wilcoxon signed-rank test
  - 3집단 이상 → Kruskal-Wallis test

### Step 3: 등분산검정

**모든 독립표본 비교(두 집단, 세 집단 이상)에서 반드시 수행한다.**

- **방법**: Levene's test
- **해석**: p < 0.05이면 등분산 가정 기각 → 이분산 보고

등분산 검정 결과에 따라 자동/수동으로 적절한 검정 방법을 선택한다:
- **t-test**: 등분산 기각 시 Welch's t-test 적용 (스크립트 옵션 활용)
- **ANOVA**: 등분산 기각 시 Welch's ANOVA (Alexander-Govern / Games-Howell) 권장

### Step 3.5: 대응표본 데이터 형식 확인 (Paired t-test인 경우)

대응표본 데이터는 두 가지 형식이 가능하다:

**Wide format (기본/권장)**: 한 행이 한 대상자, 사전·사후가 각각 별도 컬럼
```
학생ID, 사전_점수, 사후_점수
S01,    55.0,     62.3
S02,    48.2,     51.7
```
→ `--dv 사전_점수 --iv2 사후_점수` 로 지정

**Long format**: 한 행이 한 측정, 시점 구분 컬럼 존재 + (선택) ID 컬럼
```
ID,   측정시점, 점수
S01,  사전,    55.0
S01,  사후,    62.3
```
→ `--dv 점수 --iv 측정시점 --id_col ID` 로 지정
(ID 컬럼이 없으면 데이터 정렬 순서에 의존하므로 위험함)

> **참고**: Paired t-test의 Cohen's d는 `Post - Pre` (변화량) 기준으로 계산된다.

### Step 4: 본 검정 실시

검정 유형에 따라 `scripts/run_analysis.py`를 실행한다.
환경변수 `SKILL_DIR` 또는 `OUTPUT_DIR`을 활용한다.

**독립표본 / ANOVA:**
```bash
python scripts/run_analysis.py \
  --data <파일경로> \
  --dv <종속변수명> \
  --iv <독립변수명> \
  --test_type <independent|anova> \
  --alternative <two-sided|greater|less> \
  --equal_var <true|false> \
  --output_dir <출력경로>
```

**대응표본 (Wide format — 기본):**
```bash
python scripts/run_analysis.py \
  --data <파일경로> \
  --dv <사전측정 컬럼명> \
  --iv2 <사후측정 컬럼명> \
  --test_type paired \
  --alternative <two-sided|greater|less> \
  --output_dir <출력경로>
```

**대응표본 (Long format):**
```bash
python scripts/run_analysis.py \
  --data <파일경로> \
  --dv <측정값 컬럼명> \
  --iv <시점구분 컬럼명> \
  --id_col <ID 컬럼명> \
  --test_type paired \
  --alternative <two-sided|greater|less> \
  --output_dir <출력경로>
```

### Step 5: 분석표 출력 및 결과 해석

스크립트 실행 결과(`analysis_result.json`)를 바탕으로 사용자에게 요약 정보를 제공한다.

#### t-test
- 통계량(t), 자유도(df), 유의확률(p)
- 효과크기(Cohen's d)
- 95% 신뢰구간 (각 집단 평균)
- 평균 차이의 95% 신뢰구간

#### ANOVA
- 통계량(F), 유의확률(p)
- 효과크기(η²)
- 사후검정 결과 (Tukey, Scheffé, Games-Howell 등)

### Step 6: 시각화
- 집단별 정규분포 곡선 차트 (SD 기반)
- 차트 파일은 `output_dir` 및 `output_dir/report`에 저장됨

### Step 7: 결과 보고서 자동 저장
분석 스크립트는 분석 완료 후 `output_dir/report/` 폴더에 **`YYYYMMDD-result.md`** 파일을 자동 생성한다.
이 파일에는 기술통계, 정규성 검정 결과, 본 검정 결과, 차트가 모두 포함된다.
사용자에게 "분석 상세 결과가 리포트 폴더에 저장되었습니다"라고 안내한다.

## 필수 Python 패키지

- pandas, numpy
- scipy (ttest_ind, ttest_rel, f_oneway, levene, shapiro 등)
- statsmodels (pairwise_tukeyhsd)
- pingouin (welch_anova, pairwise_gameshowell) — 권장 (없으면 scipy 대체 기능 사용)
- matplotlib

## 스크립트 참조

- `scripts/run_analysis.py` — 메인 분석 실행 및 리포트 생성
- `references/report_template.md` — 리포트 템플릿 구조
- `references/posthoc_guide.md` — 사후검정 방법 가이드
