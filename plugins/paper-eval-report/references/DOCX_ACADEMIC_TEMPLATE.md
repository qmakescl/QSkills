# 학술 논문 평가 보고서 — Word 문서 생성 가이드

Phase 5에서 Word 문서를 생성할 때, 에이전트는 직접 JavaScript 코드를 작성할 필요 없이, 평가 결과를 JSON 형식으로 정리한 후 이미 제공된 `generate_report.js` 스크립트를 호출합니다.

---

## 1. 평가 데이터 JSON 스키마 준비

평가 결과를 아래 구조의 `eval_data.json` 파일로 저장합니다.

```json
{
  "paper_title": "논문 제목",
  "study_design": "SR / RCT / OBS-COHORT 등",
  "reporting_guideline_name": "PRISMA 2020",
  "quality_tool_name": "AMSTAR 2",
  "reporting_eval": [
    {
      "id": "1",
      "item": "Title",
      "verdict": "Yes",
      "findings": "초록과 제목에 명시됨",
      "suggestions": ""
    }
  ],
  "quality_eval": [
    {
      "id": "2",
      "item": "문헌 선택 독립 수행 여부",
      "critical": true,
      "verdict": "Partial Yes",
      "findings": "독립적 수행 언급은 있으나 불일치 해결 방법 없음",
      "suggestions": "불일치 시 합의 절차 명시 필요"
    }
  ],
  "grade_eval": [
    {
      "outcome": "심혈관 사망률",
      "studies": "5개 (n=1200)",
      "risk_of_bias": "0",
      "inconsistency": "-1",
      "indirectness": "0",
      "imprecision": "0",
      "publication_bias": "0",
      "certainty": "Moderate",
      "effect_estimate": "RR 0.85 (0.7-1.05)"
    }
  ],
  "overall_summary": "총평 및 권고사항 서술 내용..."
}
```

> **판정(verdict) 허용 값**: `Yes`, `Partial Yes`, `No`, `N/A` (색상 매핑용)

---

## 2. 보고서 문서 생성 실행

JSON 파일이 준비되면 다음 명령으로 스크립트를 실행합니다.

```bash
# JSON 파일을 인자로 전달하여 Word 문서 생성
node references/generate_report.js eval_data.json
```

정상 실행 시 현재 디렉토리에 `[논문제목일부]_평가보고서_YYYYMMDD.docx` 형태로 파일이 생성됩니다. 생성된 파일은 `mcp__cowork__present_files`로 사용자에게 전달합니다.
