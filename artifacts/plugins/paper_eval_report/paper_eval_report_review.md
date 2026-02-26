# `paper-eval-report` 플러그인 구조 평가 및 토큰 절감 분석

## 1. 전체 구조 요약

```
plugins/skills/paper-eval-report/
├── SKILL.md                          (166줄, 5.3KB)
└── references/
    ├── GUIDELINE_SELECTOR.md         (217줄, 9.4KB)
    ├── GRADE.md                      (223줄, 8.7KB)
    ├── DOCX_ACADEMIC_TEMPLATE.md     (244줄, 6.9KB)
    └── guidelines/                   (13개 파일, ~57KB)
        ├── PRISMA_2020.md            (4.7KB)
        ├── CONSORT_2010.md           (5.4KB)
        ├── STROBE.md                 (4.7KB)
        ├── STARD_2015.md             (5.3KB)
        ├── CARE_2013.md              (4.3KB)
        ├── AMSTAR2.md                (3.8KB)
        ├── RoB2.md                   (4.0KB)
        ├── ROBINS-I.md               (4.6KB)
        ├── ROBIS.md                  (4.1KB)
        ├── QUADAS-2.md               (3.8KB)
        ├── Newcastle-Ottawa_Scale.md (4.5KB)
        ├── PRISMA-P_2015.md          (3.8KB)
        └── PRISMA-ScR_2018.md        (3.9KB)
```

| 구성 요소 | 파일 수 | 총 크기 | 추정 토큰 |
|-----------|--------|---------|----------|
| SKILL.md | 1 | 5.3KB | ~1.8K |
| Core references (3개) | 3 | 24.9KB | ~8.3K |
| Guidelines (13개) | 13 | 56.9KB | ~19K |
| **합계** | **17** | **87.1KB** | **~29K** |

> [!IMPORTANT]
> 최악의 경우(SR + 복합 설계) 한 번의 평가에 SKILL.md + GUIDELINE_SELECTOR + GRADE + DOCX_TEMPLATE + 지침 3~4개를 모두 로딩하면 **~18K 토큰**을 reference 로딩에만 소비합니다.

---

## 2. 구조 평가: 잘 된 점 ✅

### 2-1. Lazy Loading 설계

Phase 2에서 연구 설계에 맞는 지침 파일만 선택적으로 읽는 구조가 잘 설계되어 있습니다. 전체 13개 지침을 모두 로드하지 않고, 설계별로 1~3개만 읽는 것은 토큰 절감의 핵심입니다.

### 2-2. 5단계 파이프라인 분리

Phase 1(식별) → Phase 2(로딩) → Phase 3(판정) → Phase 4(GRADE) → Phase 5(Word 출력) 순서가 명확하고, 각 Phase 간 의존성이 잘 정의되어 있습니다.

### 2-3. 사용자 확인 게이트

Phase 1-4에서 연구 설계를 식별한 후 사용자 확인을 거치도록 한 것은 잘못된 지침 적용을 방지하여 토큰 낭비를 줄이는 효과적인 설계입니다.

### 2-4. DOCX 생성 전용 템플릿

`DOCX_ACADEMIC_TEMPLATE.md`에 JavaScript 코드 스니펫을 구체적으로 포함하여 에이전트가 코드를 생성할 때 일관성을 갖추도록 한 것은 좋은 접근입니다.

---

## 3. 토큰 과다 소비 지점 및 절감 방안

### 🔴 문제 1: `GUIDELINE_SELECTOR.md` 매 실행마다 전체 로딩 (~3.1K 토큰)

**현재:** Phase 1에서 매번 217줄 전체를 로딩합니다. 그러나 **섹션 3의 키워드 사전**(77~138줄, 62줄)은 의사결정 트리(28~71줄)와 매핑 표(157~169줄)를 이미 구현한 에이전트가 일일이 참조할 필요가 낮습니다.

**절감 방안:**

- 키워드 사전(섹션 3)을 별도 파일 `KEYWORD_DICT.md`로 분리
- `GUIDELINE_SELECTOR.md`는 의사결정 트리 + 매핑 표 + 복합 설계 규칙만 유지 (~150줄로 축소)
- 의사결정 트리만으로 판별이 어려운 경우에만 키워드 사전을 추가 로딩
- **예상 절감: ~0.7K 토큰/회**

---

### 🔴 문제 2: `DOCX_ACADEMIC_TEMPLATE.md` 코드 중복 (~2.3K 토큰)

**현재:** 244줄 중 **150줄 이상이 JavaScript 코드 블록**입니다. 스타일 정의(24~53줄), 표지(60~84줄), 판정 색상 셀(100~118줄), 헤더 행(124~141줄), 준수율 표(171~183줄), 각주(191~204줄), import(211~218줄) 등이 모두 코드 예시로 작성되어 있습니다.

**문제점:**

1. 에이전트가 매번 이 코드를 읽고 → 기반으로 전체 스크립트를 다시 생성하므로, **입력 + 출력 양쪽 모두 토큰이 낭비**됨
2. 코드 스니펫들이 함수 단위로 파편화되어 있어 에이전트가 통합 시 오류 가능성 높음

**절감 방안:**

- JavaScript 코드를 **완성된 템플릿 스크립트** (`generate_report.js`)로 분리
- `DOCX_ACADEMIC_TEMPLATE.md`는 문서 구성과 규칙만 서술 (코드 제거, ~80줄로 축소)
- 에이전트는 스크립트의 데이터 부분(평가 결과 JSON)만 채워 실행
- **예상 절감: ~4K 토큰/회** (입력 1.5K + 출력 생성 코드 2.5K)

---

### 🟡 문제 3: `GRADE.md` SR 아닌 논문에서도 로딩 가능성

**현재:** SKILL.md Phase 2에서 "SR/메타분석인 경우 추가로 `references/GRADE.md`도 읽는다"라고 명시되어 있으나, `GRADE.md` (223줄, ~2.9K 토큰)는 SR이 아닌 경우에는 불필요합니다.

**상태:** Lazy loading 규칙은 제대로 정의되어 있으나, Phase 4에서 "SR/메타분석만 실행"이라는 조건이 한 줄로만 명시되어 에이전트가 놓칠 수 있습니다.

**절감 방안:**

- Phase 2 매핑 표에 GRADE.md 로딩 조건을 명시적으로 `SR` 행에만 포함
- 별도의 "GRADE 행"을 제거하고 SR 행에 통합 표기
- **예상 절감: SR 아닌 경우 ~2.9K 토큰/회**

---

### 🟡 문제 4: 개별 지침 파일 내 서술적 설명 과다

**현재:** 13개 지침 파일이 각각 3.8~5.4KB로, 항목 목록 외에 배경 설명, 적용 팁, 판정 예시 등 서술적 내용이 상당 부분 포함되어 있습니다.

**절감 방안:**

- 각 지침 파일을 **체크리스트 표 중심**으로 리팩토링 (배경 설명 축소)
- 형식: `| 번호 | 항목 | 세부 요구사항 | 판정 기준 |`
- 서술적 설명은 `guidelines/extended/` 하위 디렉터리로 분리 (필요 시만 참조)
- **예상 절감: 파일당 ~30%, 전체 ~6K 토큰**

---

### 🟢 문제 5: SKILL.md 자체의 중복 서술

**현재:** SKILL.md가 Phase 1~5를 설명하면서 references 파일 내용을 부분적으로 반복 기술하고 있습니다:

- Phase 3 판정 기준 표(98~103줄) → 각 지침 파일에도 동일 내용
- Phase 4 GRADE 순서(116~121줄) → GRADE.md와 중복
- Phase 2 매핑 표(76~86줄) → GUIDELINE_SELECTOR.md 섹션 5와 동일

**절감 방안:**

- SKILL.md에서 references 파일 내용과 중복되는 부분은 **참조 지시만** 남기기
- 예: "Phase 3 판정 기준은 각 지침 파일의 Judgment Criteria 섹션을 따른다"
- **예상 절감: ~0.5K 토큰/회**

---

## 4. 토큰 절감 종합 요약

| 절감 항목 | 현재 추정 토큰 | 절감 후 추정 | 절감량 |
|-----------|-------------|------------|-------|
| GUIDELINE_SELECTOR 분리 | 3.1K | 2.4K | **-0.7K** |
| DOCX_TEMPLATE → 스크립트 분리 | 2.3K(입력) + α(출력) | 0.8K + 0 | **-4K** |
| GRADE.md 조건부 로딩 강화 | 2.9K (SR 외 불필요) | 0 | **-2.9K** |
| 지침 파일 체크리스트화 | ~19K (전체) | ~13K | **-6K** |
| SKILL.md 중복 제거 | 1.8K | 1.3K | **-0.5K** |
| **합계** | | | **~14K 절감** |

> [!TIP]
> 가장 효과가 큰 2가지는 **DOCX 템플릿의 스크립트 분리**와 **지침 파일 체크리스트화**입니다. 이 두 가지만으로도 ~10K 토큰을 절감할 수 있습니다.

---

## 5. 기존 Antigravity 스킬과의 비교

| 항목 | `skills/paper-evaluator` (Antigravity) | `plugins/…/paper-eval-report` (Claude) |
|------|---------------------------------------|---------------------------------------|
| 최종 출력 | Markdown 채팅 출력 | **Word(.docx) 파일 생성** |
| DOCX 템플릿 | 없음 (`report_template.md`만) | `DOCX_ACADEMIC_TEMPLATE.md` (6.9KB) |
| Phase 5 | Markdown 보고서 | JavaScript로 .docx 생성 |
| 나머지 Phase 1~4 | 동일 구조 | 동일 구조 |
| 공유 references | `GUIDELINE_SELECTOR.md`, `GRADE.md`, `guidelines/` | 동일 (파일 내용 동일) |

> 두 스킬은 Phase 1~4가 사실상 동일하며, Phase 5(출력 형식)만 다릅니다. references 파일들(`GUIDELINE_SELECTOR.md`, `GRADE.md`, `guidelines/`)도 완전히 동일한 내용입니다.

---

## 6. 추가 권고사항

### 6-1. 세션 경로 하드코딩 제거

SKILL.md 148번줄의 `/sessions/adoring-serene-lamport/` 경로와 DOCX_TEMPLATE의 240번줄 경로가 특정 세션 ID에 하드코딩되어 있습니다. 이를 변수나 상대경로로 변경해야 재사용성이 확보됩니다.

### 6-2. `npm install -g docx` → 로컬 설치 권장

전역 설치(`-g`)는 권한 문제를 일으킬 수 있으므로 `npm install docx`(로컬)로 변경을 권장합니다.

### 6-3. report_template.md 부재

Antigravity 스킬에는 있는 `report_template.md`가 이 플러그인에는 없습니다. DOCX_ACADEMIC_TEMPLATE가 이를 대체하지만, Markdown 중간 보고서 구조(Phase 3~4 결과 정리)를 위한 템플릿도 있으면 에이전트의 중간 사고 과정이 줄어들어 토큰이 절감됩니다.
