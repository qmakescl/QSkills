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
│   ├── hwpx/                  # 한글(HWP/HWPX) 문서 처리 스킬
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   ├── references/
│   │   └── evals/
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

| 스킬                   | 설명                                                                                         |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| `df-basic-stats`       | DataFrame 기초 통계 자동 산출 (타입 추론, 기술통계, ydata-profiling)                         |
| `mean-comparison-test` | 평균비교 검정 (독립표본 t-test, 대응표본 t-test, ANOVA 및 사후검정)                          |
| `paper-evaluator`      | 논문 연구 설계 자동 식별 및 국제 보고/질 평가 지침 기반 평가, GRADE 등급화 (Markdown 보고서) |
| `hwpx`                 | 한글(HWP/HWPX) 문서 읽기·생성·편집 및 PDF/DOCX 변환 (대한민국 공공기관·기업 표준 문서 포맷)  |

## 플러그인 목록

| 플러그인            | 설명                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
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

```bash
npx skills add qmakescl/QSkills/skills --skill hwpx
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

### hwpx

**한글 문서 처리** — `.hwp`(구형 바이너리) 및 `.hwpx`(신형 XML 기반, KS X 6101) 파일 읽기·생성·편집, HWP ↔ PDF/DOCX 변환

> **개발 현황**: 12 Phase + v3 품질 개선(2026-03-21)을 거쳐 개발 완료. 버그 17종 발견·수정. 상세 내역은 [`dcos/hwpx.skills-process-report.md`](docs/hwpx.skills-process-report.md) 참조.

#### 트리거 키워드

`.hwp`, `.hwpx`, `한글 문서`, `hwp 파일`, `한컴오피스`, `HWP 텍스트 추출`, `한글 문서 생성`, `HWP 변환` 등

#### 요청 예시

**HWP/HWPX 파일 읽기**
> "이 hwp 파일 내용을 읽어줘"
> "보도자료.hwpx 내용을 요약해줘"

**HWPX 파일 편집**
> "보도자료.hwpx 파일에서 담당자 연락처 부분을 수정해줘"

**새 한글 문서 생성**
> "사업 계획서를 hwpx 파일로 만들어줘"

**파일 변환**
> "이 hwp 파일을 PDF로 변환해줘"

#### 실행 모델 (Subagent Execution Model)

이 스킬은 **격리된 Subagent**에 작업을 위임하는 방식으로 동작합니다.
오케스트레이터(부모 에이전트)는 중간 과정을 사용자에게 노출하지 않고 최종 결과(파일 경로 또는 추출 텍스트)만 반환합니다.

Subagent는 실행 전 반드시 스킬 디렉토리로 이동 후 스크립트를 실행합니다(`cd {Skill directory}`).

#### 자동 생성 결과물

- 추출된 텍스트
  - `.hwp` → `pyhwp` (`hwp5txt` 모듈, 임시 파일 방식)
  - `.hwpx` → `python-hwpx TextExtractor` 전용 경로 (확장자 분기 처리됨)
- 편집된 `.hwpx` 파일 (unpack → XML 직접 수정 → pack)
- 새로 생성된 `.hwpx` 파일 (`python-hwpx HwpxDocument.new()` 기반, 한컴오피스에서 정상 열림)
- 변환된 PDF 또는 DOCX 파일 (LibreOffice 래퍼)

#### 알려진 한계

| 항목             | 현황                                                                                                                                |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Bold 텍스트      | `python-hwpx` upstream 버그로 미지원. 제목 스타일로 대체                                                                            |
| 표(TABLE) 생성   | `python-hwpx` 직접 생성 미지원. unpack → XML 삽입(`hp:tbl/tr/tc`) → pack 방식으로 가능. `references/hwpx-xml.md`에 완성형 패턴 수록 |
| 이미지 삽입      | `create.py` 미구현                                                                                                                  |
| HWPX → PDF 변환  | LibreOffice HWPX 지원 미흡으로 제한적                                                                                               |
| `.hwp` 직접 쓰기 | 바이너리 포맷이므로 불가. HWPX로 생성 후 변환 안내                                                                                  |

#### 의존성

```bash
# python-hwpx - .hwpx 생성·읽기용 (핵심, Linux 호환)
pip install python-hwpx --break-system-packages

# pyhwp - .hwp 바이너리 읽기용
pip install pyhwp --break-system-packages

# LibreOffice (파일 변환, 시스템 기본 설치)
/usr/bin/libreoffice
```

> **주의**: `pyhwpx`는 Windows 전용(Win32 COM 의존)이므로 Linux/Mac 환경에서는 `python-hwpx`를 사용한다.

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
