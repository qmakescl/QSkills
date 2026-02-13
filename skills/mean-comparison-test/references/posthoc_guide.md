# 사후검정(Post-hoc) 방법 가이드

## 개요

ANOVA에서 귀무가설이 기각되면(F검정이 유의하면), 
구체적으로 어떤 집단 간에 차이가 있는지 확인하기 위해 사후검정을 실시한다.
이 스킬에서는 등분산 가정 충족 여부에 따라 적절한 사후검정 방법을 제시한다.

## 방법별 특징

### 1. Tukey HSD (Honestly Significant Difference)

- 모든 집단 쌍의 평균을 비교하는 가장 널리 사용되는 방법
- **등분산 가정 필요**
- 집단 크기가 같을 때 (balanced design) 가장 적합
- studentized range distribution 사용
- 신뢰구간(CI)도 함께 제공

### 2. Scheffé

- 가장 보수적인 사후검정 방법 (검정력이 낮음, 즉 차이를 발견하기 어려움)
- 모든 가능한 대비(contrast)에 대해 제1종 오류를 통제
- 집단 크기가 다를 때(unbalanced design)에도 적합
- **등분산 가정 필요**
- F분포 사용

### 3. Duncan의 다중범위검정 (Duncan's Multiple Range Test)

- Tukey보다 검정력이 높지만 제1종 오류 통제가 느슨함
- 집단을 동질 부분집합(homogeneous subsets)으로 분류
- **주의**: 현재 Python 표준 라이브러리(scipy, statsmodels)에서 Duncan 검정을 직접 지원하지 않음
- **대안**: 이 스킬에서는 **Bonferroni 보정을 적용한 Pairwise t-test**를 대안으로 수행하여 보수적이고 안전한 결과를 제공함

### 4. Dunnett's T3 / Games-Howell (이분산)

- **등분산 가정이 불필요**한 사후검정 방법
- 이분산 상황에서 가장 적합
- Levene 검정에서 등분산 가정이 기각되었을 때 사용
- 이 스킬에서는 `pingouin` 패키지가 설치된 경우 **Games-Howell** 검정을 우선 수행하며, 그렇지 않은 경우 **Bonferroni-corrected Welch t-test**를 대안으로 사용 (Dunnett's T3 근사)
- JSON 결과 키: `games_howell_or_fallback`

## 선택 가이드

| 상황 | 권장 방법 |
|------|-----------|
| 등분산 + 동일 표본 크기 | **Tukey HSD** |
| 등분산 + 다른 표본 크기 | **Scheffé** |
| 탐색적 연구 (덜 보수적) | Duncan (단, 대안 방법 사용 시 주의) |
| 이분산 (등분산 가정 기각) | **Games-Howell / Bonferroni-Welch** |
| 엄격한 판단 필요 | Scheffé |

## 이 스킬의 접근법

이 스킬에서는 가능한 모든 방법을 수행하여 결과를 제시한다.

1. **등분산 충족 시**: Tukey HSD 결과를 주된 해석 근거로 삼는다.
2. **등분산 미충족 시**: Games-Howell (또는 Bonferroni-Welch 대안) 결과를 주된 해석 근거로 삼는다.
3. **결과 불일치 시**: 보수적인 방법(Scheffé, Bonferroni)의 결과를 신뢰하는 것이 안전하다.
