# 타입 추론 알고리즘 상세

## 목차
1. [추론 흐름도](#추론-흐름도)
2. [판정 기준 상세](#판정-기준-상세)
3. [80% 임계값의 근거](#80-임계값의-근거)
4. [알려진 한계](#알려진-한계)

---

## 추론 흐름도

```
열(Series) 입력
    │
    ├─ datetime64 dtype? ───── YES → "datetime"
    │
    ├─ integer dtype? ──────── YES → "integer"
    │   (numpy int8~64, pandas Int8~64)
    │
    ├─ float dtype? ────────── YES → "continuous"
    │
    └─ object dtype? ──────── YES → 2차 추론
         │
         ├─ dropna 후 비어있음? → "categorical"
         │
         ├─ pd.to_numeric 변환
         │   └─ 80%+ 성공?
         │       ├─ 모두 정수(x%1==0)? → "integer"
         │       └─ 소수점 포함?       → "continuous"
         │
         └─ pd.to_datetime 변환
             └─ 80%+ 성공? → "datetime"
                 └─ 실패   → "categorical"
```

## 판정 기준 상세

### 1차 판정: pandas dtype 기반

| dtype 범주 | 포함 타입 | 분류 결과 |
|-----------|----------|----------|
| `datetime64[ns]`, `datetime64[ns, tz]` | 모든 datetime64 변형 | `datetime` |
| `int8` ~ `int64`, `uint8` ~ `uint64` | numpy 정수 | `integer` |
| `Int8` ~ `Int64`, `UInt8` ~ `UInt64` | pandas nullable 정수 | `integer` |
| `float16` ~ `float64` | 모든 float | `continuous` |
| `object`, `string`, `category`, `bool` | 나머지 | 2차 추론 대상 |

### 2차 판정: 값 기반 파싱 (object 열 전용)

object 열은 pandas가 자동 추론하지 못한 경우이므로, 실제 값을 파싱하여 재판정한다.

**숫자 파싱 (`pd.to_numeric`)**
- 결측 제외 후 `pd.to_numeric(series, errors="coerce")` 수행
- 변환 성공 비율 = `notna().sum() / len(non_null)`
- 80% 초과 시 숫자형으로 판정
  - 변환된 값 전체가 정수(`x % 1 == 0`)면 → `integer`
  - 소수점이 하나라도 있으면 → `continuous`

**날짜 파싱 (`pd.to_datetime`)**
- 숫자 판정에 실패한 경우에만 시도
- `pd.to_datetime(series, errors="coerce", infer_datetime_format=True)` 수행
- 변환 성공 비율 80% 초과 시 → `datetime`

## 80% 임계값의 근거

- 실무 데이터는 일부 오입력(typo), 특수 코드("N/A", "-", "미응답" 등)가 섞여 있음
- 100%로 설정하면 오입력 1건으로 전체 열이 categorical로 판정됨
- 50%는 너무 관대하여 혼합열을 숫자로 오판할 위험
- 80%는 "대부분 숫자/날짜"인 열을 올바르게 인식하면서도 혼합열은 거르는 실용적 기준
- 필요 시 `compute_stats.py`의 `NUMERIC_THRESHOLD` 상수를 조정 가능

## 알려진 한계

| 상황 | 현재 동작 | 개선 방향 |
|------|----------|----------|
| "1", "2", "3" 같은 코드값 (범주인데 숫자) | integer로 판정 | unique 비율이 낮으면 categorical 우선 (향후) |
| 통화 문자열 ("₩50,000") | categorical | 전처리 파이프라인 별도 추가 |
| 혼합 타임존 날짜 | 파싱 실패 가능 | utc=True 옵션 시도 (향후) |
| boolean 열 | categorical | bool 전용 분류 추가 가능 |
