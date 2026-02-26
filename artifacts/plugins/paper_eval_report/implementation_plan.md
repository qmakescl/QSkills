# Paper-Eval-Report 토큰 절감 구현 계획

분석 보고서에서 식별한 5가지 토큰 과다 소비 지점을 해결합니다. 예상 총 절감량은 **~14K 토큰/회**입니다.

## Proposed Changes

### 개선 1: GUIDELINE_SELECTOR.md 키워드 사전 분리

#### [MODIFY] [GUIDELINE_SELECTOR.md](file:///Users/yoonani/Works/QSkills/plugins/skills/paper-eval-report/references/GUIDELINE_SELECTOR.md)

- 섹션 3 "설계별 상세 키워드 사전" (75~138줄) 제거
- 대신 "필요 시 `KEYWORD_DICT.md` 참조" 한 줄 추가
- 나머지 (의사결정 트리, 매핑 표, 복합 설계 규칙, 사용자 프롬프트) 유지

#### [NEW] [KEYWORD_DICT.md](file:///Users/yoonani/Works/QSkills/plugins/skills/paper-eval-report/references/KEYWORD_DICT.md)

- 기존 섹션 3의 키워드 사전 내용을 그대로 이동
- 의사결정 트리로 판별이 어려운 경우에만 에이전트가 추가 로딩

---

### 개선 2: DOCX_ACADEMIC_TEMPLATE → 스크립트 분리

#### [NEW] [generate_report.js](file:///Users/yoonani/Works/QSkills/plugins/skills/paper-eval-report/references/generate_report.js)

- 기존 DOCX_ACADEMIC_TEMPLATE.md의 JavaScript 코드 스니펫들을 하나의 완성된 Node.js 스크립트로 통합
- 평가 데이터를 JSON 파일(`eval_data.json`)에서 읽어 Word 문서 생성
- 에이전트는 JSON 데이터만 생성하면 되므로 코드 생성 토큰 절감

#### [MODIFY] [DOCX_ACADEMIC_TEMPLATE.md](file:///Users/yoonani/Works/QSkills/plugins/skills/paper-eval-report/references/DOCX_ACADEMIC_TEMPLATE.md)

- JavaScript 코드 블록 전체 제거
- 문서 구성 규칙, 판정 색상 규칙, JSON 데이터 스키마만 서술 (~80줄로 축소)

---

### 개선 3: 13개 지침 파일 체크리스트 중심 리팩토링

#### [MODIFY] 모든 `guidelines/*.md` (13개 파일)

각 파일에서:

- 상단 메타 정보(정식 명칭, 발표 연도, 항목 수, 참고 문헌, 사이트) → **2줄 요약**으로 축소
- "개요" 섹션 → **제거** (SKILL.md에서 이미 맥락 제공)
- "참고", "주의사항", "확장판" 등 부가 섹션 → **제거**
- **체크리스트 표만 유지** + 판정 기준이 있으면 간결한 표로 유지
- 예상 축소: 파일당 ~30%

---

### 개선 4: SKILL.md 정리 (중복 제거 + 경로/설치 수정)

#### [MODIFY] [SKILL.md](file:///Users/yoonani/Works/QSkills/plugins/skills/paper-eval-report/references/../SKILL.md)

- Phase 2 매핑 표: GUIDELINE_SELECTOR.md와 중복 → "GUIDELINE_SELECTOR.md 섹션 5 참조" 지시로 대체
- Phase 3 판정 기준 표: "각 지침 파일의 판정 기준을 따른다"로 대체
- Phase 5: 스크립트 파일(`generate_report.js`) 실행 흐름으로 변경, JSON 스키마 기술
- 세션 경로 하드코딩 → `$WORKSPACE/` 변수 사용
- `npm install -g docx` → `npm install docx` (로컬 설치)
- `pip install ... --break-system-packages` → 가상환경 내 설치 안내

---

## Verification Plan

### 수동 검증

- `wc -c`로 변경 전후 파일 크기(bytes) 비교하여 절감률 계산
- 전체 references 디렉터리 크기 비교
- `generate_report.js`가 유효한 Node.js 문법인지 `node --check generate_report.js`로 확인
- 각 지침 파일이 체크리스트 표를 온전히 유지하고 있는지 목시 확인
