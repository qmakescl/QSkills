---
description: 논문 PDF를 평가하고 학술 Word 보고서 생성
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, Task
argument-hint: [논문-파일-경로]
---

논문 파일 `$ARGUMENTS`를 평가하고 학술 양식 Word 보고서(.docx)를 생성한다.

## 실행 순서

### Step 1. 파일 확인

`$ARGUMENTS`가 주어진 경우 해당 경로의 파일을 사용한다.
주어지지 않은 경우: 워크스페이스 폴더(mnt/Research/)에서 PDF 또는 Word 파일을 찾아 사용자에게 선택을 요청한다.

### Step 2. paper-eval-report 스킬 파이프라인 실행

paper-eval-report 스킬의 5단계 파이프라인을 전체 실행한다:

1. **Phase 1** — 논문 텍스트 추출 후 연구 설계 식별, 사용자 확인
2. **Phase 2** — 연구 설계에 맞는 지침 파일 선택적 로딩
3. **Phase 3** — 항목별 준수 판정 (Yes / Partial / No / N/A)
4. **Phase 4** — GRADE 등급화 (SR/메타분석만)
5. **Phase 5** — 학술 양식 Word 문서 생성 및 저장

### Step 3. 결과 전달

생성된 `.docx` 파일을 `mcp__cowork__present_files`로 사용자에게 전달한다.
채팅에는 핵심 수치(준수율, GRADE 등급)만 간략히 요약한다.

## 출력 파일 위치

`[워크스페이스]/[논문파일명]_평가보고서_YYYYMMDD.docx`
