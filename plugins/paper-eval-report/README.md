# paper-eval-report

논문 PDF를 받아 국제 보고 지침으로 자동 평가하고, 학술 양식 Word 보고서(.docx)로 출력하는 플러그인.

---

## 개요

기존 `paper-evaluator` 스킬이 Markdown 보고서를 생성하는 데서 한 걸음 더 나아가,
이 플러그인은 평가 결과를 **학술 양식 Word 문서(.docx)**로 자동 변환해 전달합니다.

지원하는 평가 지침:
- **보고 지침**: PRISMA 2020, CONSORT 2010, STROBE, STARD 2015, CARE 2013, PRISMA-ScR, PRISMA-P
- **질 평가 도구**: AMSTAR 2, RoB 2.0, ROBINS-I, QUADAS-2, Newcastle-Ottawa Scale
- **근거 확실성**: GRADE (SR/메타분석)

---

## 컴포넌트

| 컴포넌트 | 이름 | 역할 |
|----------|------|------|
| Skill | `paper-eval-report` | 자연어 요청 시 자동 트리거 — 전체 파이프라인 실행 |
| Command | `/evaluate-paper` | 단일 논문 평가 → Word 보고서 생성 |
| Command | `/paper-dashboard` | 여러 논문 평가 결과 → 비교 요약 대시보드 생성 |

---

## 설치 요구 사항

플러그인 실행 시 아래 패키지가 자동으로 확인·설치됩니다:

```bash
pip install pdfplumber python-docx --break-system-packages
npm install -g docx
```

Node.js가 설치되어 있지 않은 경우 Word 생성 단계에서 안내가 제공됩니다.

---

## 사용법

### 방법 1: 자연어로 요청

```
이 논문 평가해서 Word 보고서로 만들어줘
```
```
논문 평가 결과를 docx로 받고 싶어
```
```
이 PDF PRISMA 준수 여부 확인하고 보고서 생성해줘
```

### 방법 2: 슬래시 커맨드

```
/evaluate-paper paper.pdf
```

논문 파일 경로 없이 입력하면 워크스페이스에서 자동으로 파일을 탐색합니다.

### 여러 논문 요약 대시보드

```
/paper-dashboard
```

워크스페이스에 있는 모든 평가 보고서를 자동으로 수집하여 비교 요약 Word 문서를 생성합니다.

---

## 출력 파일

| 커맨드 | 출력 파일 |
|--------|----------|
| `/evaluate-paper` | `[논문파일명]_평가보고서_YYYYMMDD.docx` |
| `/paper-dashboard` | `paper_dashboard_YYYYMMDD.docx` |

---

## Word 보고서 구성

생성되는 `.docx` 파일은 아래 구조로 작성됩니다:

1. **표지** — 논문 제목, 평가일, 적용 지침 버전
2. **논문 기본 정보** — 저자, 학술지, DOI, 연구 설계
3. **적용된 평가 지침** — 지침명 및 항목 수
4. **보고 지침 준수 평가** — 항목별 ✅/⚠️/❌ 색상 체크리스트 표
5. **방법론적 질 / 비뚤림 위험 평가** — 도구별 항목 평가 표
6. **GRADE 근거 확실성** — Evidence Profile 표 (SR/메타분석만)
7. **종합 요약 및 개선 권고사항** — 강점, 우선 개선 항목, 권고 조치 표

---

## 버전 정보

- 플러그인 버전: `0.1.0`
- 참조 지침 버전: PRISMA 2020, AMSTAR 2 (2017), QUADAS-2 (2011), GRADE
