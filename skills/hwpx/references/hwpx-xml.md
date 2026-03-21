# HWPX XML 구조 참조 (section0.xml 기반)

HWPX는 KS X 6101 기반의 XML 문서 포맷이다.
본문은 `Contents/section0.xml`, 스타일·폰트 정의는 `Contents/header.xml`에 있다.

> **경고**: `content.hml`은 구형 비표준 파일이다. 편집·생성 대상 본문 파일은 반드시 `section0.xml`이다.

## 목차
1. [네임스페이스](#네임스페이스)
2. [section0.xml 전체 구조](#section0xml-전체-구조)
3. [header.xml 구조 (스타일·폰트)](#headerxml-구조)
4. [단락(p) 구조](#단락p-구조)
5. [표(tbl) 구조](#표tbl-구조)
6. [이미지(pic) 구조](#이미지pic-구조)
7. [주요 속성값 단위](#주요-속성값-단위)
8. [자주 쓰는 패턴](#자주-쓰는-패턴)

---

## 네임스페이스

실제 python-hwpx 생성 파일 기준 (모두 **2011**, 태그 **소문자**):

```xml
<!-- section0.xml 루트 선언 -->
xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"
xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"

<!-- header.xml 루트 선언 (주요 접두사) -->
xmlns:hh="http://www.hancom.co.kr/hwpml/2011/head"
xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"
xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"
xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core"
```

> **절대 금지**: `2012` 네임스페이스, 대문자 태그(`P/RUN/T/TABLE`) 사용 금지.
> HML 포맷(`content.hml`, 2012 네임스페이스)과 혼동하지 않는다.

---

## section0.xml 전체 구조

```xml
<?xml version="1.0" encoding="UTF-8"?>
<hs:sec xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"
        xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section">

  <!-- 첫 번째 hp:p — hp:secPr 필수 (없으면 "파일 손상" 오류) -->
  <hp:p id="3121190098" paraPrIDRef="0" styleIDRef="0"
        pageBreak="0" columnBreak="0" merged="0">
    <hp:run charPrIDRef="0">
      <hp:secPr textDirection="HORIZONTAL" spaceColumns="1134"
                tabStop="8000" outlineShapeIDRef="1" masterPageCnt="0">
        <hp:pagePr landscape="WIDELY" width="59528" height="84186"
                   gutterType="LEFT_ONLY">
          <hp:margin header="4252" footer="4252" gutter="0"
                     left="8504" right="8504" top="5668" bottom="4252"/>
        </hp:pagePr>
      </hp:secPr>
    </hp:run>
    <hp:run charPrIDRef="0">
      <hp:t/>
    </hp:run>
  </hp:p>

  <!-- 이후 본문 단락들 -->
  <hp:p id="2380727813" paraPrIDRef="0" styleIDRef="2"
        pageBreak="0" columnBreak="0" merged="0">
    <hp:run charPrIDRef="0">
      <hp:t>문서 제목</hp:t>
    </hp:run>
  </hp:p>

</hs:sec>
```

### 핵심 규칙

- **id 속성**: 큰 난수 정수 사용. 문서 내에서 유일해야 한다. 새 `<hp:p>` 추가 시 기존 최대값 +1 또는 임의 큰 수 사용.
- **`<hp:secPr>`** 는 반드시 첫 번째 `<hp:p>`의 `<hp:run>` 안에 위치한다.
- **`<hp:t>` 안에서 `\n` 사용 금지** — 새 줄은 새 `<hp:p>` 블록으로 표현.
- **네임스페이스 선언 제거 금지** — `<hs:sec>` 루트의 xmlns 선언을 유지한다.

---

## header.xml 구조

`header.xml`에는 `section0.xml`에서 IDRef로 참조하는 스타일·폰트가 정의된다.
BODY만 편집할 때는 수정 불필요 (기존 IDRef 재사용).

### 스타일 목록 (styleIDRef 대응표)

```xml
<hh:style id="0"  name="바탕글"  engName="Normal"    paraPrIDRef="0"  charPrIDRef="0"/>
<hh:style id="1"  name="본문"    engName="Body"      paraPrIDRef="1"  charPrIDRef="0"/>
<hh:style id="2"  name="개요 1"  engName="Outline 1" paraPrIDRef="2"  charPrIDRef="0"/>
<hh:style id="3"  name="개요 2"  engName="Outline 2" paraPrIDRef="3"  charPrIDRef="0"/>
<hh:style id="4"  name="개요 3"  engName="Outline 3" paraPrIDRef="4"  charPrIDRef="0"/>
```

`<hp:p styleIDRef="2">` → 개요 1(제목) 스타일 적용.

### charPr (글자 모양)

```xml
<hh:charPr id="0" height="1000" textColor="#000000" ...>
  <hh:fontRef hangul="1" latin="1" hanja="1" .../>
  <!-- height: 1/100 pt 단위. 1000 = 10pt, 1200 = 12pt -->
</hh:charPr>
```

### paraPr (단락 모양)

```xml
<hh:paraPr id="0" tabPrIDRef="0" condense="0" textDir="LTR">
  <hh:align horizontal="JUSTIFY" vertical="BASELINE"/>
</hh:paraPr>
```

---

## 단락(p) 구조

```xml
<hp:p id="2380727813" paraPrIDRef="0" styleIDRef="0"
      pageBreak="0" columnBreak="0" merged="0">

  <!-- RUN: 같은 글자 모양의 텍스트 묶음 -->
  <hp:run charPrIDRef="0">
    <hp:t>텍스트 내용</hp:t>
  </hp:run>

  <!-- 같은 단락 내 다른 글자 모양 -->
  <hp:run charPrIDRef="1">
    <hp:t>굵은 텍스트</hp:t>
  </hp:run>

</hp:p>
```

### 빈 단락 (빈 줄)

```xml
<hp:p id="2956786762" paraPrIDRef="0" styleIDRef="0"
      pageBreak="0" columnBreak="0" merged="0">
  <hp:run charPrIDRef="0"><hp:t/></hp:run>
</hp:p>
```

### XML 특수문자 이스케이프

| 원문자 | 이스케이프 |
|--------|-----------|
| `&` | `&amp;` |
| `<` | `&lt;` |
| `>` | `&gt;` |

---

## 표(tbl) 구조

표도 `hp:` 네임스페이스를 사용한다. `<hp:tbl>`은 `<hp:run>` 안에 위치한다.

```xml
<hp:p id="1000000001" paraPrIDRef="0" styleIDRef="0"
      pageBreak="0" columnBreak="0" merged="0">
  <hp:run charPrIDRef="0">
    <hp:tbl id="1" colCount="3" rowCount="2"
            cellSpacing="0" borderFillIDRef="1"
            numMergedCell="0" zOrder="0"
            numberingType="NONE" textWrap="SQUARE"
            textFlow="BOTH_SIDES" lock="0">

      <!-- 행 -->
      <hp:tr>
        <!-- 셀: colAddr/rowAddr는 0부터 시작 -->
        <hp:tc name="" header="0" hasMargin="0" protect="0"
               editable="0" dirty="0" borderFillIDRef="2">
          <hp:tcPr colSpan="1" rowSpan="1"
                   colAddr="0" rowAddr="0"
                   width="14400" height="1000">
            <hp:tcMar left="141" right="141" top="0" bottom="0"/>
          </hp:tcPr>
          <!-- 셀 내부는 일반 hp:p와 동일 -->
          <hp:p id="1000000002" paraPrIDRef="0" styleIDRef="0"
                pageBreak="0" columnBreak="0" merged="0">
            <hp:run charPrIDRef="0">
              <hp:t>셀 내용</hp:t>
            </hp:run>
          </hp:p>
        </hp:tc>
        <!-- 나머지 셀 ... -->
      </hp:tr>
      <!-- 나머지 행 ... -->

    </hp:tbl>
  </hp:run>
</hp:p>
```

### 표 태그 요약

| 태그 | 역할 | 주요 속성 |
|------|------|----------|
| `<hp:tbl>` | 표 전체 | `colCount`, `rowCount`, `cellSpacing`, `borderFillIDRef` |
| `<hp:tr>` | 행 | — |
| `<hp:tc>` | 셀 | `borderFillIDRef` |
| `<hp:tcPr>` | 셀 크기/위치 | `colAddr`, `rowAddr`, `colSpan`, `rowSpan`, `width`, `height` |
| `<hp:tcMar>` | 셀 여백 | `left`, `right`, `top`, `bottom` (HWP 단위) |

> `borderFillIDRef="1"` → 테두리 없음, `borderFillIDRef="2"` → 기본 채우기(셀 배경).
> header.xml의 `<hh:borderFill>` id와 매칭된다.

---

## 이미지(pic) 구조

이미지는 `Contents/BinData/` 폴더에 저장 후 section0.xml에서 참조한다.

```xml
<hp:p id="2000000001" paraPrIDRef="0" styleIDRef="0"
      pageBreak="0" columnBreak="0" merged="0">
  <hp:run charPrIDRef="0">
    <hp:pic>
      <!-- width/height: HWP 단위 (1/7200 inch) -->
      <hp:shapeComponent id="0" zOrder="0" groupLevel="0"
                         width="14400" height="10800">
        <hp:image href="BinData/image001.png" binDataRef="0"
                  type="link"/>
      </hp:shapeComponent>
    </hp:pic>
  </hp:run>
</hp:p>
```

BinData 파일을 `Contents/BinData/image001.png`로 복사한 후 위 XML을 삽입한다.

---

## 주요 속성값 단위

| 속성 | 단위 | 예시 |
|------|------|------|
| 글자 크기 (`height`) | 1/100 pt | `1000` = 10pt, `1200` = 12pt |
| 길이 (여백, 너비 등) | HWP 단위 (1/7200 inch ≈ 0.00353mm) | `14400` ≈ 50.8mm (2인치) |
| 페이지 너비 A4 | HWP 단위 | `59528` (210mm) |
| 페이지 높이 A4 | HWP 단위 | `84186` (297mm) |
| 셀 기본 여백 | HWP 단위 | `141` ≈ 0.5mm |

---

## 자주 쓰는 패턴

### 본문 단락 (기본 스타일)

```xml
<hp:p id="9999000001" paraPrIDRef="0" styleIDRef="0"
      pageBreak="0" columnBreak="0" merged="0">
  <hp:run charPrIDRef="0">
    <hp:t>본문 내용입니다.</hp:t>
  </hp:run>
</hp:p>
```

### 제목 단락 (개요 1, styleIDRef="2")

```xml
<hp:p id="9999000002" paraPrIDRef="0" styleIDRef="2"
      pageBreak="0" columnBreak="0" merged="0">
  <hp:run charPrIDRef="0">
    <hp:t>1장 제목</hp:t>
  </hp:run>
</hp:p>
```

### 3열 2행 표 (헤더행 + 데이터행)

```xml
<hp:p id="9999000010" paraPrIDRef="0" styleIDRef="0"
      pageBreak="0" columnBreak="0" merged="0">
  <hp:run charPrIDRef="0">
    <hp:tbl id="1" colCount="3" rowCount="2" cellSpacing="0"
            borderFillIDRef="1" numMergedCell="0" zOrder="0"
            numberingType="NONE" textWrap="SQUARE"
            textFlow="BOTH_SIDES" lock="0">
      <hp:tr>
        <hp:tc name="" header="0" hasMargin="0" protect="0"
               editable="0" dirty="0" borderFillIDRef="2">
          <hp:tcPr colSpan="1" rowSpan="1" colAddr="0" rowAddr="0"
                   width="14173" height="1000">
            <hp:tcMar left="141" right="141" top="0" bottom="0"/>
          </hp:tcPr>
          <hp:p id="9999000011" paraPrIDRef="0" styleIDRef="0"
                pageBreak="0" columnBreak="0" merged="0">
            <hp:run charPrIDRef="0"><hp:t>제품명</hp:t></hp:run>
          </hp:p>
        </hp:tc>
        <hp:tc name="" header="0" hasMargin="0" protect="0"
               editable="0" dirty="0" borderFillIDRef="2">
          <hp:tcPr colSpan="1" rowSpan="1" colAddr="1" rowAddr="0"
                   width="14173" height="1000">
            <hp:tcMar left="141" right="141" top="0" bottom="0"/>
          </hp:tcPr>
          <hp:p id="9999000012" paraPrIDRef="0" styleIDRef="0"
                pageBreak="0" columnBreak="0" merged="0">
            <hp:run charPrIDRef="0"><hp:t>수량</hp:t></hp:run>
          </hp:p>
        </hp:tc>
        <hp:tc name="" header="0" hasMargin="0" protect="0"
               editable="0" dirty="0" borderFillIDRef="2">
          <hp:tcPr colSpan="1" rowSpan="1" colAddr="2" rowAddr="0"
                   width="14174" height="1000">
            <hp:tcMar left="141" right="141" top="0" bottom="0"/>
          </hp:tcPr>
          <hp:p id="9999000013" paraPrIDRef="0" styleIDRef="0"
                pageBreak="0" columnBreak="0" merged="0">
            <hp:run charPrIDRef="0"><hp:t>단가</hp:t></hp:run>
          </hp:p>
        </hp:tc>
      </hp:tr>
      <hp:tr>
        <hp:tc name="" header="0" hasMargin="0" protect="0"
               editable="0" dirty="0" borderFillIDRef="2">
          <hp:tcPr colSpan="1" rowSpan="1" colAddr="0" rowAddr="1"
                   width="14173" height="1000">
            <hp:tcMar left="141" right="141" top="0" bottom="0"/>
          </hp:tcPr>
          <hp:p id="9999000014" paraPrIDRef="0" styleIDRef="0"
                pageBreak="0" columnBreak="0" merged="0">
            <hp:run charPrIDRef="0"><hp:t>사과</hp:t></hp:run>
          </hp:p>
        </hp:tc>
        <hp:tc name="" header="0" hasMargin="0" protect="0"
               editable="0" dirty="0" borderFillIDRef="2">
          <hp:tcPr colSpan="1" rowSpan="1" colAddr="1" rowAddr="1"
                   width="14173" height="1000">
            <hp:tcMar left="141" right="141" top="0" bottom="0"/>
          </hp:tcPr>
          <hp:p id="9999000015" paraPrIDRef="0" styleIDRef="0"
                pageBreak="0" columnBreak="0" merged="0">
            <hp:run charPrIDRef="0"><hp:t>10</hp:t></hp:run>
          </hp:p>
        </hp:tc>
        <hp:tc name="" header="0" hasMargin="0" protect="0"
               editable="0" dirty="0" borderFillIDRef="2">
          <hp:tcPr colSpan="1" rowSpan="1" colAddr="2" rowAddr="1"
                   width="14174" height="1000">
            <hp:tcMar left="141" right="141" top="0" bottom="0"/>
          </hp:tcPr>
          <hp:p id="9999000016" paraPrIDRef="0" styleIDRef="0"
                pageBreak="0" columnBreak="0" merged="0">
            <hp:run charPrIDRef="0"><hp:t>500</hp:t></hp:run>
          </hp:p>
        </hp:tc>
      </hp:tr>
    </hp:tbl>
  </hp:run>
</hp:p>
```
