---
name: paper-eval-report
description: >
  논문 파일(PDF, Word, 텍스트)을 받아 연구 설계를 자동 식별하고,
  PRISMA·CONSORT·STROBE·STARD·CARE·AMSTAR 2·RoB 2.0·ROBINS-I·QUADAS-2·NOS 등
  국제 보고·질 평가 지침을 자동 적용한 뒤, GRADE 근거 확실성 등급화까지 수행하여
  학술 양식 Word 문서(.docx)로 최종 보고서를 출력하는 스킬.
  사용자가 "논문 평가", "논문 검토", "Word 보고서", "평가 결과를 Word로",
  "논문 평가해서 docx로", "PRISMA 확인", "CONSORT 준수 여부",
  "GRADE 평가", "비뚤림 위험" 등의 표현을 사용하면 반드시 이 스킬을 사용한다.
version: 0.1.0
---

# Paper Evaluation → Academic Word Report

논문 PDF를 평가하고 학술 양식 Word 문서로 출력하는 5단계 파이프라인.

## 필요 패키지

플러그인 루트 경로(`plugins/paper-eval-report/references`)에서:

```bash
npm install docx
```

---

## Phase 1: 연구 설계 식별

### 1-1. 참조 규칙 로딩

`references/GUIDELINE_SELECTOR.md`를 Read 도구로 읽어 식별 규칙을 숙지한다.

### 1-2. 논문 텍스트 추출

```python
import pdfplumber
with pdfplumber.open("paper.pdf") as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
```

DOCX인 경우:

```python
from docx import Document
doc = Document("paper.docx")
text = "\n".join(p.text for p in doc.paragraphs)
```

### 1-3. 연구 설계 판정

`GUIDELINE_SELECTOR.md`의 의사결정 트리를 순서대로 적용한다:
1차 키워드(직접 명시) → 2차 키워드(방법론 단서) → 구조적 단서.

### 1-4. 사용자 확인 (필수)

```
📋 연구 설계 식별 결과

확인된 설계: [설계명]
식별 근거:
  - "[키워드/표현]"

적용할 평가 지침:
  - 보고 지침: [지침명]
  - 질/비뚤림 평가: [도구명]
  - GRADE 적용: [Yes / No]

계속 진행할까요?
```

---

## Phase 2: 지침 파일 선택적 로딩

설계 확정 후 아래 표에 따라 필요한 파일만 Read 도구로 로딩한다.

| 설계 코드 | 읽을 파일 |
|-----------|----------|
| SR | PRISMA_2020.md + AMSTAR2.md |
| SR-P | PRISMA-P_2015.md |
| SR-SCR | PRISMA-ScR_2018.md |
| RCT | CONSORT_2010.md + RoB2.md |
| OBS-COHORT | STROBE.md + NOS.md |
| OBS-CC | STROBE.md + NOS.md |
| OBS-CS | STROBE.md + NOS.md |
| DX | STARD_2015.md + QUADAS-2.md |
| CR | CARE_2013.md |

SR + 진단 정확도 복합 설계: PRISMA_2020 + AMSTAR2 + QUADAS-2 + GRADE.md

SR/메타분석: 추가로 `GRADE.md` 로딩.

---

## Phase 3: 항목별 준수 판정

논문 각 섹션을 순서대로 읽으며 지침 항목별 판정:

| 판정 | 기준 |
|------|------|
| ✅ Yes | 완전하고 명확하게 기술 |
| ⚠️ Partial | 부분 기술, 일부 요소 누락 |
| ❌ No | 전혀 기술 없음 |
| ➖ N/A | 해당 설계에 적용 불가 |

각 항목에 대해 판정 근거(섹션/문장), 미충족 내용, 개선 제안을 기록한다.

준수율 계산:

- 적용 항목 수 = 전체 - N/A
- 준수율 (%) = (Yes + Partial) / 적용 항목 × 100
- 엄격 준수율 (%) = Yes / 적용 항목 × 100

---

## Phase 4: GRADE 등급화 (SR/메타분석만)

`references/GRADE.md`를 참고하여 각 결과(Outcome)별로:

1. 출발점 설정 (RCT → High, 관찰연구 → Low)
2. 5개 하향 도메인 순서 판정 (비뚤림 위험 / 비일관성 / 비직접성 / 비정밀성 / 출판편향)
3. 상향 기준 확인 (관찰연구 기반 SR만)
4. 최종 등급 산출: High / Moderate / Low / Very Low

---

## Phase 5: 학술 양식 Word 보고서 생성

평가 결과를 기반으로 `paper-eval-report/references/` 경로에서 JSON 구조를 구성하고 Node.js 스크립트를 통해 Word 문서(.docx)를 생성합니다.

### 스크립트 실행 순서

1. 평가 결과를 `report_data.json` 포맷으로 작성(스키마는 `DOCX_ACADEMIC_TEMPLATE.md` 참조)하여 `references` 폴더 내 저장.
2. `references` 폴더에서 `node generate_report.js` 실행
3. `Evaluation_Report.docx` 생성 확인
4. 출력 파일(`Evaluation_Report.docx`)을 사용자 작업 폴더 등 지정된 곳으로 이동.

---

## 오류 처리

- PDF 추출 실패: OCR 필요 여부 안내 후 pytesseract 시도
- 연구 설계 불명확: GUIDELINE_SELECTOR.md 섹션 6 프롬프트로 사용자 확인
- docx 라이브러리 오류: `npm install docx` 재실행 후 재시도
- Node.js 미설치: `apt-get install nodejs -y` 안내
