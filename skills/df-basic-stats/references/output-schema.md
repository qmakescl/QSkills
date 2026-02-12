# 출력 JSON 스키마 정의

## 최상위 구조

```json
{
  "dataframe_shape": {
    "rows": "<int>",
    "cols": "<int>"
  },
  "columns": ["<ColumnStats>"],
  "profile_report": "<string|null>",
  "_warning": "<string|null>"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `dataframe_shape` | object | 행·열 수 |
| `columns` | array | 각 열의 통계 객체 리스트 |
| `profile_report` | string \| null | 생성된 HTML 리포트 경로. 미생성 시 null |
| `_warning` | string \| null | 빈 데이터프레임 등 경고 메시지 |

---

## ColumnStats 공통 필드

모든 타입에 포함되는 필드:

```json
{
  "column_name": "<string>",
  "inferred_type": "continuous | integer | datetime | categorical",
  "total_count": "<int>",
  "valid_count": "<int>",
  "missing_count": "<int>",
  "missing_rate": "<float 0~1>"
}
```

---

## 타입별 추가 필드

### continuous

```json
{
  "mean": "<float|null>",
  "std": "<float|null>",
  "median": "<float|null>",
  "min": "<float|null>",
  "max": "<float|null>"
}
```

### integer

continuous의 모든 필드 + mode:

```json
{
  "mean": "<float|null>",
  "std": "<float|null>",
  "median": "<float|null>",
  "min": "<float|null>",
  "max": "<float|null>",
  "mode": "<int|float|null>"
}
```

### datetime

```json
{
  "min": "<string|null>",
  "max": "<string|null>",
  "mode": "<string|null>"
}
```

날짜값은 ISO 8601 문자열로 직렬화된다.

### categorical

고유값이 **8개 이하**인 경우 (분포 시각화 포함):

```json
{
  "unique_count": "<int>",
  "mode": "<string|null>",
  "distribution": {
    "<value>": {"count": "<int>", "ratio": "<float 0~1>"}
  },
  "distribution_description": "<string>",
  "distribution_chart": "<string|null>",
  "distribution_chart_base64": "<string|null>"
}
```

| 필드 | 설명 |
|------|------|
| `distribution` | 전체 범주의 건수 및 비율 |
| `distribution_description` | 분포를 자연어로 요약한 텍스트 |
| `distribution_chart` | 막대 차트 PNG 파일 경로 (미생성 시 null) |
| `distribution_chart_base64` | 차트의 base64 인코딩 (matplotlib 미설치 시 null) |

고유값이 **9개 이상**인 경우 (기존 방식):

```json
{
  "unique_count": "<int>",
  "mode": "<string|null>",
  "top5_values": {
    "<value>": "<count>"
  }
}
```

---

## 전체 예시

```json
{
  "dataframe_shape": {"rows": 500, "cols": 4},
  "columns": [
    {
      "column_name": "revenue",
      "inferred_type": "continuous",
      "total_count": 500,
      "valid_count": 487,
      "missing_count": 13,
      "missing_rate": 0.026,
      "mean": 125430.55,
      "std": 45230.12,
      "median": 118200.0,
      "min": 1200.0,
      "max": 350000.0
    },
    {
      "column_name": "employee_count",
      "inferred_type": "integer",
      "total_count": 500,
      "valid_count": 500,
      "missing_count": 0,
      "missing_rate": 0.0,
      "mean": 42.8,
      "std": 25.3,
      "median": 38.0,
      "min": 1.0,
      "max": 200.0,
      "mode": 25
    },
    {
      "column_name": "region",
      "inferred_type": "categorical",
      "total_count": 500,
      "valid_count": 498,
      "missing_count": 2,
      "missing_rate": 0.004,
      "unique_count": 5,
      "mode": "서울",
      "top5_values": {"서울": 180, "부산": 120, "대구": 90, "인천": 70, "광주": 38}
    },
    {
      "column_name": "created_at",
      "inferred_type": "datetime",
      "total_count": 500,
      "valid_count": 500,
      "missing_count": 0,
      "missing_rate": 0.0,
      "min": "2023-01-01 00:00:00",
      "max": "2024-06-30 00:00:00",
      "mode": "2023-03-15 00:00:00"
    }
  ],
  "profile_report": "report.html"
}
```
