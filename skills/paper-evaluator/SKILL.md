---
name: paper-evaluator
description: >
  논문 파일(PDF, Word, 텍스트)을 받아 연구 설계를 자동으로 식별한 뒤,
  PRISMA·CONSORT·STROBE·STARD·CARE·AMSTAR 2·RoB 2.0·ROBINS-I·QUADAS-2·NOS 등
  국제 보고·질 평가 지침을 자동 적용하고, SR/메타분석의 경우 GRADE 근거 확실성
  등급화까지 수행하여 Markdown 보고서를 출력하는 논문 평가 에이전트 스킬.
  논문 심사, 투고 전 자가 점검, 연구 질 평가, 비뚤림 위험 평가 등 모든 상황에서
  사용한다. 사용자가 "논문 평가", "논문 검토", "체크리스트 평가", "PRISMA 확인",
  "CONSORT 준수 여부", "GRADE 평가", "비뚤림 위험", "systematic review 질 평가"
  등의 표현을 사용하면 반드시 이 스킬을 사용한다.
---

# Paper Evaluator — 논문 평가 에이전트

## 스킬 개요

이 스킬은 5단계 파이프라인으로 논문을 평가한다.

1. Phase 1 — 초록(Abstract) 기반 연구 설계 자동 식별
2. Phase 2 — 설계에 맞는 평가 지침 파일만 선택적 로딩 (lazy loading)
3. Phase 3 — 항목별 상세 준수 판정 (Yes / Partial Yes / No / N/A)
4. Phase 4 — GRADE 근거 확실성 등급화 (SR/메타분석 전용)
5. Phase 5 — Markdown 보고서 출력

참조 문서 구조:
- `references/GUIDELINE_SELECTOR.md` — 초록 분석 엔진 및 설계→지침 매핑
- `references/GRADE.md` — GRADE 등급화 기준
- `references/report_template.md` — Phase 5 최종 보고서 템플릿
- `references/guidelines/` — 개별 지침 파일 13개

---

## 필요 패키지 (처음 실행 시 설치)

실행 전 다음 명령을 Bash로 실행한다.

    pip install pdfplumber pypdf python-docx --break-system-packages -q

---

## Phase 1: 연구 설계 식별

### 1-1. 참조 규칙 로딩

먼저 `references/GUIDELINE_SELECTOR.md`를 Read 도구로 읽어 식별 규칙을 숙지한다.

### 1-2. 논문 텍스트 추출

파일 형식에 따라 아래 방식으로 전체 텍스트를 추출한다.

PDF 처리:

    import pdfplumber
    with pdfplumber.open("paper.pdf") as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

DOCX 처리:

    from docx import Document
    doc = Document("paper.docx")
    text = "\n".join(p.text for p in doc.paragraphs)

텍스트/Markdown은 그대로 활용한다.

### 1-3. 연구 설계 판정

`GUIDELINE_SELECTOR.md`의 의사결정 트리와 키워드 사전을 적용한다.
1차 키워드(직접 명시) → 2차 키워드(방법론 단서) → 구조적 단서 순서로 판정한다.
복합 설계는 복합 설계 처리 규칙을 참고한다.

### 1-4. 사용자 확인 (필수)

설계 식별 후 반드시 아래 형식으로 사용자에게 보고하고 확인을 받는다.

    📋 연구 설계 식별 결과

    확인된 설계: [설계명]
    식별 근거:
      - "[초록에서 발견된 키워드/표현]"
      - "[추가 단서]"

    적용할 평가 지침:
      - 보고 지침: [지침명]
      - 질/비뚤림 평가: [도구명]
      - GRADE 적용: [Yes / No]

    계속 진행할까요? (다른 설계라면 알려주세요)

사용자가 수정을 요청하면 해당 설계로 재설정 후 진행한다.

---

## Phase 2: 지침 파일 선택적 로딩 (Lazy Loading)

연구 설계가 확정된 후, 해당 설계에 필요한 파일만 Read 도구로 읽는다.
불필요한 파일은 절대 읽지 않는다. 이는 컨텍스트 크기를 줄이고 API 오류를 방지하기 위함이다.

| 설계 코드 | 읽을 파일 |
|-----------|----------|
| SR | guidelines/PRISMA_2020.md + guidelines/AMSTAR2.md |
| SR-P | guidelines/PRISMA-P_2015.md |
| SR-SCR | guidelines/PRISMA-ScR_2018.md |
| RCT | guidelines/CONSORT_2010.md + guidelines/RoB2.md |
| OBS-COHORT | guidelines/STROBE.md + guidelines/Newcastle-Ottawa_Scale.md |
| OBS-CC | guidelines/STROBE.md + guidelines/Newcastle-Ottawa_Scale.md |
| OBS-CS | guidelines/STROBE.md + guidelines/Newcastle-Ottawa_Scale.md |
| DX | guidelines/STARD_2015.md + guidelines/QUADAS-2.md |
| CR | guidelines/CARE_2013.md |

SR이고 진단 정확도 연구를 포함하는 복합 설계라면 PRISMA_2020.md + AMSTAR2.md + QUADAS-2.md를 읽는다.
SR/메타분석인 경우 추가로 `references/GRADE.md`도 읽는다.

지침 파일을 읽은 뒤 각 지침의 항목 목록과 판정 기준을 내부 체크리스트로 구성한다.

---

## Phase 3: 항목별 상세 준수 판정

논문의 각 섹션(제목 → 초록 → 서론 → 방법 → 결과 → 토의 → 기타)을 순서대로 읽으며
지침의 각 항목이 요구하는 내용이 논문에 기술되어 있는지 확인한다.

판정 기준은 다음과 같다.

- Yes (✅): 해당 항목이 논문에 완전하고 명확하게 기술됨
- Partial Yes (⚠️): 부분적으로 기술됨, 일부 요소가 빠져 있거나 불충분
- No (❌): 해당 항목이 전혀 기술되지 않음
- N/A (➖): 해당 연구 설계에 적용 불가한 항목

각 항목에 대해 다음을 기록한다.
- 판정 근거: 논문의 어느 섹션/문장에서 확인했는지 (없으면 "기술 없음")
- 미충족 내용: Partial Yes / No인 경우 무엇이 구체적으로 누락됐는지
- 개선 제안: No / Partial Yes 항목에 대한 보완 방법

준수율 계산 공식:
- 전체 적용 항목 수 = 전체 항목 수 - N/A 항목 수
- 준수율 (%) = (Yes + Partial Yes) / 전체 적용 항목 수 x 100
- 엄격 준수율 (%) = Yes / 전체 적용 항목 수 x 100

---

## Phase 4: GRADE 근거 확실성 등급화

SR / 메타분석으로 확정된 경우에만 실행한다. 다른 설계는 이 단계를 건너뛴다.
`references/GRADE.md`는 Phase 2에서 이미 읽었으므로, 그 내용을 바탕으로 평가한다.

아래 순서로 평가한다.

1. 결과(Outcome) 목록 확인: SR에서 평가한 주요 결과를 논문에서 확인하고, 결과마다 별도 평가한다.
2. 출발점 설정: 포함 연구가 RCT면 High, 관찰연구면 Low에서 출발한다.
3. 5개 하향 도메인 순서대로 판정한다.
   - 비뚤림 위험: SR 내 개별 연구 비뚤림 평가 결과 참조
   - 비일관성: I2 수치, 신뢰구간 중첩 여부
   - 비직접성: PICO 요소와 임상 질문의 일치 여부
   - 비정밀성: OIS 충족 여부, CI 폭
   - 출판 편향: 깔때기 그림, Egger's test 결과
4. 상향 기준 확인 (관찰연구 기반 SR만): 큰 효과 크기, 용량-반응 관계, 반대 방향 교란
5. 최종 등급 산출: High / Moderate / Low / Very Low 중 하나로 결정한다.

---

## Phase 5: Markdown 보고서 출력

`references/report_template.md`를 Read 도구로 읽어 보고서 구조를 확인한다.
해당 템플릿에 맞춰 논문 평가 결과를 채워서 최종 보고서를 채팅에 출력한다.
보고서는 전체를 Markdown 표 형식으로 구성하여 가독성을 확보한다.

---

## 오류 처리 및 예외 상황

파일 읽기 실패 시:
- PDF 텍스트 추출 실패 시 OCR 필요 여부를 사용자에게 안내한다.
- 스캔 PDF는 pytesseract로 OCR 시도 후 재추출한다.

연구 설계 불명확 시:
- GUIDELINE_SELECTOR.md 섹션 6의 프롬프트 템플릿으로 사용자 확인을 요청한다.
- 사용자 응답 기반으로 재진행한다.

초록 없는 논문:
- 제목 + 서론 + 방법 첫 부분 기반으로 판정한다.
- 사용자에게 "초록이 없어 제목과 방법론 섹션 기반으로 설계를 식별했습니다"라고 안내한다.

복합 설계 논문:
- GUIDELINE_SELECTOR.md 섹션 4의 복합 설계 처리 규칙을 적용한다.
- 복수 지침 병행 적용 시 각 지침별로 섹션을 구분하여 보고한다.
