# QSkills

AI Agent Skills Collection — Q가 AI Agent의 도움을 받아 만들고 있는 Agent Skills 입니다.

## 프로젝트 구조

```
QSkills/
├── skills/                    # 스킬 모음
│   ├── df-basic-stats/        # DataFrame 기초 통계 분석 스킬
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   ├── references/
│   │   └── evals/
│   ├── mean-comparison-test/  # 평균비교 검정 스킬
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   ├── references/
│   │   └── evals/
│   ├── paper-evaluator/       # 논문 평가 스킬 (마크다운)
│   │   ├── SKILL.md
│   │   └── references/
├── plugins/                   # 플러그인 모음
│   └── paper-eval-report/     # 논문 평가 및 Word 보고서 생성 플러그인
│       ├── README.md
│       ├── plugin.json
│       ├── commands/
│       ├── skills/
│       └── references/
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
| `paper-evaluator` | 논문 연구 설계 자동 식별 및 국제 보고/질 평가 지침 기반 평가, GRADE 등급화 (Markdown 보고서) |

## 플러그인 목록

| 플러그인 | 설명 |
|----------|------|
| `paper-eval-report` | 논문 연구 설계 식별 및 각종 지침 기반 평가 산출물을 학술 양식 Word 문서(.docx)로 자동 변환 생성 |

## 스킬 설치

[skills](https://skills.sh)를 통해 개별 스킬을 설치할 수 있습니다.

> **사전 요구사항**: skills 실행을 위해 [Node.js](https://nodejs.org/)가 설치되어 있어야 합니다.

```bash
npx skills add qmakescl/QSkills/skills --skill mean-comparison-test
```

```bash
npx skills add qmakescl/QSkills/skills --skill df-basic-stats
```

```bash
npx skills add qmakescl/QSkills/skills --skill paper-evaluator
```

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

### paper-evaluator

**논문 평가** — 연구 설계 파악, 보고 지침(PRISMA, CONSORT 등) 기반 판정 및 GRADE 근거 확실성 평가

#### 트리거 키워드

`논문 평가`, `논문 검토`, `체크리스트 평가`, `PRISMA 확인`, `CONSORT 준수 여부`, `GRADE 평가`, `비뚤림 위험`, `systematic review 질 평가` 등

#### 요청 예시

**일반 논문 평가**
> "이 논문 PDF 파일을 평가해줘"

**특정 지침 기반 평가**
> "첨부한 메타분석 논문에 대해 PRISMA 지침 준수 여부와 GRADE 평가를 진행해줘"

#### 자동 생성 결과물

- 마크다운 보고서 (연구 설계 식별, 항목별 상세 준수 판정, GRADE 등급화, 결론)

---

## 플러그인 사용법

플러그인은 복합 워크플로우나 학술 문서 생성처럼 확장된 기능을 제공하며, 명확하고 빠른 실행을 위한 슬래시 커맨드(`/`)를 활용할 수 있습니다.

### paper-eval-report

**논문 평가 및 표 양식 Word 자동 생성** — `paper-evaluator` 스킬의 강력한 분석 결과를 학술 연구자 친화적인 Microsoft Word(.docx) 형식의 표와 체크리스트로 렌더링.

#### 커맨드 사용

이 플러그인은 커맨드 기반의 실행을 지원합니다.

- **개별 논문 평가**: `/evaluate-paper [논문 PDF 명]`
- **평가 결과 대시보드 생성**: `/paper-dashboard`

#### 자연어 요청 예시
>
> "이 논문 읽고 평가해서 양식에 맞춘 docx 보고서로 출력해줘"
> "기존에 만들었던 평가 보고서들 다 모아서 논문 요약 대시보드 Word 파일 하나 만들어줘"

#### 자동 생성 결과물

- 학술 표준 포맷 `.docx` 파일 (최종 평가 리포트 / 논문 요약 대시보드)
- 평가 결과 데이터 (JSON 포맷, 진행과정 기록용)

---

Q 의 지침을 받아 Antigravity 가 생성했습니다.
