# QSkills

AI Agent Skills Collection — 에이전트가 활용할 수 있는 스킬 모음 저장소입니다.

## 프로젝트 구조
```
QSkills/
├── skills/              # 스킬 모음
│   └── df-basic-stats/  # DataFrame 기초 통계 분석 스킬
│       ├── SKILL.md
│       ├── scripts/
│       ├── references/
│       └── evals/
├── pyproject.toml       # 프로젝트 설정 및 의존성
├── uv.lock              # 버전 고정 파일
└── .python-version      # Python 버전 지정
```

## 스킬 목록

| 스킬 | 설명 |
|------|------|
| `df-basic-stats` | DataFrame 기초 통계 자동 산출 (타입 추론, 기술통계, ydata-profiling) |
