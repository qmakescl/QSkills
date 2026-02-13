# QSkills

AI Agent Skills Collection — 에이전트가 활용할 수 있는 스킬 모음 저장소입니다.

## 프로젝트 구조
```
QSkills/
├── skills/                    # 스킬 모음
│   ├── df-basic-stats/        # DataFrame 기초 통계 분석 스킬
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   ├── references/
│   │   └── evals/
│   └── mean-comparison-test/  # 평균비교 검정 스킬
│       ├── SKILL.md
│       ├── scripts/
│       ├── references/
│       └── evals/
├── datasets/                  # 분석용 샘플 데이터
├── pyproject.toml             # 프로젝트 설정 및 의존성
├── uv.lock                    # 버전 고정 파일
└── .python-version            # Python 버전 지정
```

## 스킬 목록

| 스킬 | 설명 |
|------|------|
| `df-basic-stats` | DataFrame 기초 통계 자동 산출 (타입 추론, 기술통계, ydata-profiling) |
| `mean-comparison-test` | 평균비교 검정 (독립표본 t-test, 대응표본 t-test, ANOVA 및 사후검정) |

## 스킬 사용법
이 저장소의 스킬은 AI 에이전트가 자동으로 인식하고 실행합니다.
사용자는 자연어로 요청하면 에이전트가 적절한 스킬을 선택하여 분석을 수행합니다.

### mean-comparison-test
**평균비교 검정** — 독립표본 t-test, 대응표본 t-test, ANOVA 및 사후검정

#### 트리거 키워드
`평균 비교`, `t-test`, `t검정`, `분산분석`, `ANOVA`, `집단 간 차이`, `그룹별 평균`, `사후검정`, `두 집단 비교` 등

#### 요청 예시

**독립표본 t-test**
> "sample_independent.csv 데이터에서 남녀 간 시험 점수 차이가 있는지 검정해줘"

**대응표본 t-test**
> "sample_paired.csv에서 교육 프로그램 적용 후 자기효능감이 증가했는지 유의수준 0.1에서 검정해줘"

**일원배치 분산분석 (ANOVA)**
> "sample_anova.csv 데이터셋에서 운동프로그램에 따라 체중감량에 유의한 차이가 있는지 분석해줘"

**유의수준 지정**
> "유의수준 0.01로 검정해줘"

#### 자동 생성 결과물
- 마크다운 보고서 (분석 개요, 정규성/등분산 검정, 기술통계, 검정 결과, 결론, 시각화)
- JSON 분석 결과 파일
- 평균 ± 95% CI 시각화 차트
