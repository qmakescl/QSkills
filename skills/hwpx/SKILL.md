---
name: hwpx
description: |
  한글(HWP/HWPX) 문서를 읽고, 생성하고, 편집하는 스킬. 한국 문서 작성 프로그램인 한컴오피스(한글)의 파일 형식을 다룬다.
  다음 상황에서 반드시 이 스킬을 사용한다:
  - .hwp 또는 .hwpx 파일 언급 시
  - "한글 문서", "hwp 파일", "한컴오피스" 관련 요청 시
  - HWP 파일에서 텍스트/표/이미지 추출 요청 시
  - 새 한글 문서 생성 또는 기존 문서 편집 요청 시
  - HWP → PDF/DOCX 변환 또는 DOCX/PDF → HWP 변환 요청 시
  HWP, HWPX, 한글, 한컴, hwp 파일 등의 표현이 나오면 항상 이 스킬을 사용한다.
---

# HWP / HWPX 문서 처리

한글(HWP)은 대한민국 공공기관·기업에서 광범위하게 쓰이는 문서 포맷이다.
파일 형식은 두 가지다: **`.hwp`** (구형 바이너리) · **`.hwpx`** (신형 XML 기반, ZIP 아카이브).

## Quick Reference

| 작업 | 접근법 |
|------|--------|
| .hwp 텍스트 읽기 | `pyhwp` 라이브러리 사용 → `scripts/read_hwp.py` |
| .hwpx 텍스트 읽기 | `python-hwpx` TextExtractor → `scripts/read_hwp.py` |
| .hwpx 편집 | unpack → XML 수정 → pack → `scripts/pack.py` |
| 새 .hwpx 생성 | `python-hwpx` HwpxDocument.new() → `scripts/create.py` |
| HWP ↔ PDF/DOCX 변환 | LibreOffice 래퍼 → `scripts/soffice.py` |

---

## 포맷 구조

### .hwpx (신형, XML 기반) — KS X 6101 실제 구조
```
document.hwpx (ZIP)
├── mimetype                      ← "application/hwp+zip" (ZIP_STORED, 반드시 첫 번째)
├── version.xml                   ← 버전 정보
├── META-INF/
│   ├── container.xml             ← Contents/content.hpf 를 rootfile 로 지정
│   └── manifest.xml
├── Contents/
│   ├── content.hpf               ← OPF 패키지 (header + section0 목록)
│   ├── header.xml                ← 스타일, 폰트, charPr, paraPr 정의
│   └── section0.xml              ← 본문 (핵심). 첫 <hp:p>에 <hp:secPr> 필수
├── Preview/
│   └── PrvText.txt               ← 텍스트 미리보기
└── settings.xml
```

> **주의**: `content.hml` 파일은 **구 비표준 포맷**이다. 실제 한글 앱이 읽는 본문 파일은 `section0.xml`이다.

### .hwp (구형, 바이너리)
OLE Compound Document 포맷. `pyhwp` 라이브러리로만 파싱 가능.
직접 편집 불가 → 텍스트 추출 또는 LibreOffice로 hwpx/docx 변환 후 작업한다.

---

## 의존성 설치

```bash
# python-hwpx - .hwpx 생성·읽기용 (핵심 라이브러리, Linux 호환)
pip install python-hwpx --break-system-packages

# pyhwp - .hwp 바이너리 읽기용 (없으면 설치)
pip install pyhwp --break-system-packages

# pip 미설치 시
python -m pip --version || python -m ensurepip --upgrade
```

> **주의**: `pyhwpx`는 Windows 전용이므로 Linux 환경에서는 사용 불가. `python-hwpx`를 사용한다.

LibreOffice는 시스템에 기본 설치되어 있다 (`/usr/bin/libreoffice`).

---

## 텍스트 읽기

```bash
# .hwp 또는 .hwpx 모두 지원
python scripts/read_hwp.py document.hwp
python scripts/read_hwp.py document.hwpx
# → 텍스트를 stdout 또는 지정 파일로 출력
```

- `.hwpx`: `python-hwpx`의 `TextExtractor`를 우선 사용
- `.hwp`: `pyhwp`의 `hwp5txt` 모듈 사용; 없으면 LibreOffice로 변환 후 추출

**python-hwpx TextExtractor 직접 사용 (스크립트 없이):**
```python
from hwpx import TextExtractor
te = TextExtractor("document.hwpx")
text = te.extract_text()
print(text)
```

---

## .hwpx 파일 언팩/팩

```bash
# 언팩: .hwpx → 디렉토리 (XML 들여쓰기 포함)
python scripts/unpack.py document.hwpx unpacked/

# 팩: 디렉토리 → .hwpx
python scripts/pack.py unpacked/ output.hwpx
```

언팩 후 `unpacked/Contents/section0.xml`이 본문 XML이다.
텍스트는 `<hp:t>` 태그 안에 있다 (소문자). 자세한 XML 구조는 `references/hwpx-xml.md` 참조.

---

## .hwpx 편집 (3단계 순서 준수)

### Step 1: 언팩
```bash
python scripts/unpack.py document.hwpx unpacked/
```

### Step 2: XML 편집
`unpacked/Contents/section0.xml`을 직접 수정한다.
- 텍스트 수정: `<hp:t>` 태그 내용 변경 (소문자)
- 문단 추가: `<hp:p>` 블록 추가 (기존 문단 복사 후 수정 권장, 소문자)
- **Edit 도구를 직접 사용한다** (Python 스크립트 작성 불필요)

XML 구조 상세는 `references/hwpx-xml.md`를 읽는다.

### Step 3: 팩
```bash
python scripts/pack.py unpacked/ output.hwpx --original document.hwpx
```

---

## 새 .hwpx 생성

`python-hwpx` 라이브러리의 `HwpxDocument.new()`를 사용하면 한글에서 정상적으로 열리는 HWPX 파일을 생성할 수 있다.

```python
# scripts/create.py 사용 (권장)
python scripts/create.py --output new_document.hwpx

# Python에서 직접 호출
from scripts.create import HWPXDocument

doc = HWPXDocument()
doc.add_paragraph("제목", style="title")    # 개요 1 스타일
doc.add_paragraph("소제목", style="h2")     # 개요 2 스타일
doc.add_paragraph("본문 내용입니다.")        # 기본(바탕글) 스타일
doc.add_blank_line()                         # 빈 줄
doc.save("output.hwpx")
```

**스타일 이름:**

| style= | 한글 스타일 | 용도 |
|--------|------------|------|
| `"title"` / `"h1"` | 개요 1 | 문서 제목 |
| `"h2"` | 개요 2 | 소제목 |
| `"h3"` | 개요 3 | 소소제목 |
| `"body"` / `"본문"` | 본문 | 본문 텍스트 |
| `"normal"` / `"바탕글"` | 바탕글 | 기본(생략 시) |

상세 API는 `scripts/create.py` 소스 참조.

---

## 파일 변환

```bash
# HWP → PDF
python scripts/soffice.py --headless --convert-to pdf document.hwp

# HWP → DOCX
python scripts/soffice.py --headless --convert-to docx document.hwp

# HWPX → PDF
python scripts/soffice.py --headless --convert-to pdf document.hwpx
```

변환 결과는 원본 파일과 같은 디렉토리에 저장된다.

---

## XML 핵심 규칙

- **네임스페이스는 2011** — `http://www.hancom.co.kr/hwpml/2011/*` (2012 아님)
- **태그는 소문자** — `<hp:p>`, `<hp:run>`, `<hp:t>` (대문자 `P/RUN/T` 사용 금지)
- **`<hp:t>` 안에서 `\n` 사용 금지** — 새 줄은 새 `<hp:p>` 블록으로 표현
- **첫 번째 `<hp:p>` 에 `<hp:secPr>` 필수** — 없으면 한글 앱이 "파일 손상됨" 오류를 냄
- **네임스페이스 반드시 유지** — 기존 XML의 xmlns 선언을 제거하지 않는다
- **`id` 속성 중복 금지** — 새 `<hp:p>` 추가 시 기존 최대 id + 1 사용
- **본문 파일은 section0.xml** — `content.hml`은 비표준이므로 생성하지 않는다
- **BinData 경로** — 이미지 참조 시 `Contents/BinData/` 하위 파일명과 일치해야 함

XML 전체 구조, 네임스페이스, 스타일 참조 방법은 `references/hwpx-xml.md`에서 확인한다.

---

## Dependencies

- **python-hwpx**: `.hwpx` 생성 및 텍스트 추출 (Linux 호환, `pip install python-hwpx --break-system-packages`)
- **pyhwp**: `.hwp` 바이너리 텍스트 추출 (`pip install pyhwp --break-system-packages`)
- **LibreOffice**: 파일 변환 (`/usr/bin/libreoffice`)
- **Python stdlib**: `zipfile`, `xml.etree.ElementTree`, `lxml` (HWPX 직접 편집 시)
