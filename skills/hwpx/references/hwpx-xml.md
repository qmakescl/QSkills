# HWPX XML 구조 참조

HWPX는 한국 국가표준 KS X 6101 기반의 XML 문서 포맷이다.
`content.hml`이 본문의 핵심 파일이다.

## 목차
1. [네임스페이스](#네임스페이스)
2. [content.hml 전체 구조](#contenthml-전체-구조)
3. [HEAD 섹션 (스타일·폰트)](#head-섹션)
4. [BODY 섹션 (본문)](#body-섹션)
5. [단락(P) 구조](#단락p-구조)
6. [표(TABLE) 구조](#표table-구조)
7. [이미지(PICTURE) 구조](#이미지picture-구조)
8. [주요 속성값 단위](#주요-속성값-단위)

---

## 네임스페이스

```xml
xmlns:hml="http://www.hancom.co.kr/hwpml/2012/core"
xmlns:hp="http://www.hancom.co.kr/hwpml/2012/paragraph"
xmlns:hs="http://www.hancom.co.kr/hwpml/2012/section"
xmlns:hh="http://www.hancom.co.kr/hwpml/2012/head"
xmlns:hf="http://www.hancom.co.kr/hwpml/2012/fill"
xmlns:hr="http://www.hancom.co.kr/hwpml/2012/run"
xmlns:ht="http://www.hancom.co.kr/hwpml/2012/table"
xmlns:hpf="http://www.hancom.co.kr/hwpml/2012/picture"
xmlns:ha="http://www.hancom.co.kr/hwpml/2012/app"
```

---

## content.hml 전체 구조

```xml
<?xml version="1.0" encoding="UTF-8"?>
<hml:HWPMLDocType xmlns:hml="..." xmlns:hp="..." version="1.3.0.0">
  <hml:HEAD>
    <!-- 문서 속성, 스타일, 폰트, 단락모양 정의 -->
  </hml:HEAD>
  <hml:BODY>
    <hml:SECTION>
      <!-- 본문 단락들 -->
    </hml:SECTION>
  </hml:BODY>
  <hml:TAIL>
    <!-- 각주, 미주 등 (선택) -->
  </hml:TAIL>
</hml:HWPMLDocType>
```

---

## HEAD 섹션

HEAD에는 BODY에서 참조(IDRef)하는 스타일 정의가 있다.
BODY를 편집할 때는 HEAD를 수정할 필요가 없다 (기존 IDRef 재사용).

### CHARSHAPELIST (글자 모양)
```xml
<hml:CHARSHAPELIST>
  <hml:CHARSHAPE id="0"
    height="1000"     <!-- 글자 크기: 1/100 pt 단위 (1000 = 10pt) -->
    bold="0"          <!-- 굵게: 0|1 -->
    italic="0"        <!-- 기울임: 0|1 -->
    underline="0"     <!-- 밑줄: 0|1 -->
    strikeout="0">    <!-- 취소선: 0|1 -->
    <hml:FONTID face="함초롬바탕" lang="ko"/>
    <hml:FONTID face="Arial"      lang="en"/>
  </hml:CHARSHAPE>
</hml:CHARSHAPELIST>
```

### PARASHAPELIST (단락 모양)
```xml
<hml:PARASHAPELIST>
  <hml:PARASHAPE id="0"
    lineSpacing="160"           <!-- 줄 간격 (%) -->
    lineSpacingType="percent"
    spaceAbove="0"              <!-- 문단 위 여백 (1/100 mm) -->
    spaceBelow="0"              <!-- 문단 아래 여백 -->
    align="justify">            <!-- 정렬: left|center|right|justify -->
  </hml:PARASHAPE>
</hml:PARASHAPELIST>
```

### STYLELIST (스타일)
```xml
<hml:STYLELIST>
  <hml:STYLE id="0" name="바탕글"
    charPrIDRef="0"   <!-- CHARSHAPE id 참조 -->
    paraPrIDRef="0"   <!-- PARASHAPE id 참조 -->
    type="para"/>
</hml:STYLELIST>
```

---

## BODY 섹션

```xml
<hml:BODY>
  <hml:SECTION>
    <hml:SECDEF .../>   <!-- 섹션 속성 (페이지 크기, 여백 등) -->
    <hp:P id="0" paraPrIDRef="0" styleIDRef="0">
      <!-- 단락 내용 -->
    </hp:P>
    <hp:P id="1" ...>
      <!-- 다음 단락 -->
    </hp:P>
  </hml:SECTION>
</hml:BODY>
```

### SECDEF (섹션 속성)
```xml
<hml:SECDEF>
  <hml:PAGEDEF
    width="59528"      <!-- 페이지 너비: 1/100 mm (A4: 21000 → 59528 HWP 단위) -->
    height="84188"     <!-- 페이지 높이 -->
    landscape="0">     <!-- 가로: 1, 세로: 0 -->
    <hml:PAGEMARGIN
      top="5000" bottom="5000"
      left="4200" right="4200"
      header="1700" footer="1700" gutter="0"/>
  </hml:PAGEDEF>
</hml:SECDEF>
```

> **단위**: HWPX 길이 단위 = 1/7200 inch ≈ 1/100mm의 약 0.353배.
> A4 기준: 너비 210mm → 59528 HWP 단위, 높이 297mm → 84188 HWP 단위.

---

## 단락(P) 구조

```xml
<hp:P id="0"
  paraPrIDRef="0"   <!-- HEAD의 PARASHAPE id 참조 -->
  styleIDRef="0">   <!-- HEAD의 STYLE id 참조 -->

  <!-- RUN: 같은 글자 모양의 텍스트 묶음 -->
  <hp:RUN charPrIDRef="0">   <!-- HEAD의 CHARSHAPE id 참조 -->
    <hp:T>텍스트 내용</hp:T>
  </hp:RUN>

  <!-- 여러 RUN: 같은 단락 내 다른 글자 모양 -->
  <hp:RUN charPrIDRef="1">
    <hp:T>굵은 텍스트</hp:T>
  </hp:RUN>

</hp:P>
```

### 빈 단락 (빈 줄)
```xml
<hp:P id="5" paraPrIDRef="0" styleIDRef="0"/>
```

### 텍스트에서 주의할 점
- `<hp:T>` 안에서 줄바꿈 금지 — 새 줄은 새 `<hp:P>` 블록으로
- XML 특수문자 이스케이프 필수: `&` → `&amp;`, `<` → `&lt;`, `>` → `&gt;`

---

## 표(TABLE) 구조

```xml
<hp:P id="10" paraPrIDRef="0" styleIDRef="0">
  <hp:TABLE>
    <ht:TABLE
      id="0"
      numberingType="caption"
      rowCount="2"
      colCount="3"
      cellSpacing="0">

      <ht:ROW>
        <ht:CELL
          colAddr="0" rowAddr="0"
          colSpan="1" rowSpan="1">
          <hp:P id="11" paraPrIDRef="0" styleIDRef="0">
            <hp:RUN charPrIDRef="0">
              <hp:T>셀 내용</hp:T>
            </hp:RUN>
          </hp:P>
        </ht:CELL>
        <!-- 추가 셀들 -->
      </ht:ROW>

      <!-- 추가 행들 -->
    </ht:TABLE>
  </hp:TABLE>
</hp:P>
```

---

## 이미지(PICTURE) 구조

이미지는 `Contents/BinData/` 폴더에 저장 후 content.hml에서 참조한다.

```xml
<hp:P id="20" paraPrIDRef="0" styleIDRef="0">
  <hp:PICTURE>
    <hpf:PICTURE
      id="0"
      binDataRef="0"   <!-- BinData id 참조 -->
      width="14400"    <!-- 너비 (HWP 단위) -->
      height="10800">  <!-- 높이 (HWP 단위) -->
    </hpf:PICTURE>
  </hp:PICTURE>
</hp:P>
```

HEAD의 BINDATA 섹션에 파일 등록:
```xml
<hml:BINDATA>
  <hml:BINITEM id="0"
    type="file"
    inStorage="1"
    name="BinData/image001.png"/>
</hml:BINDATA>
```

---

## 주요 속성값 단위

| 속성 | 단위 | 예시 |
|------|------|------|
| 글자 크기 (`height`) | 1/100 pt | `1000` = 10pt, `1200` = 12pt |
| 길이 (여백, 너비 등) | HWP 단위 (1/7200 inch) | `14400` ≈ 2인치 |
| 줄 간격 (`lineSpacing`) | % | `160` = 160% |
| 단락 여백 (`spaceAbove/Below`) | 1/100 mm | `500` = 5mm |

---

## 자주 쓰는 패턴

### 제목 단락 (굵게, 20pt)
```xml
<!-- HEAD에 추가 -->
<hml:CHARSHAPE id="1" height="2000" bold="1" italic="0" underline="0" strikeout="0">
  <hml:FONTID face="함초롬바탕" lang="ko"/>
  <hml:FONTID face="Arial" lang="en"/>
</hml:CHARSHAPE>

<!-- BODY에서 사용 -->
<hp:P id="0" paraPrIDRef="0" styleIDRef="0">
  <hp:RUN charPrIDRef="1">
    <hp:T>문서 제목</hp:T>
  </hp:RUN>
</hp:P>
```

### 가운데 정렬 단락
```xml
<!-- HEAD에 추가 -->
<hml:PARASHAPE id="1" lineSpacing="160" lineSpacingType="percent"
               spaceAbove="0" spaceBelow="0" align="center"/>

<!-- BODY에서 사용 -->
<hp:P id="5" paraPrIDRef="1" styleIDRef="0">
  <hp:RUN charPrIDRef="0">
    <hp:T>가운데 정렬 텍스트</hp:T>
  </hp:RUN>
</hp:P>
```
