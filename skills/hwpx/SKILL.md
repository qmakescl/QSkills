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
| .hwpx 텍스트 읽기 | 언팩 후 XML 파싱 → `scripts/unpack.py` |
| .hwpx 편집 | unpack → XML 수정 → pack → `scripts/pack.py` |
| 새 .hwpx 생성 | `scripts/create.py` 참조 |
| HWP ↔ PDF/DOCX 변환 | LibreOffice 래퍼 → `scripts/soffice.py` |

---

## 포맷 구조

### .hwpx (신형, XML 기반)
```
document.hwpx (ZIP)
├── mimetype                      ← "application/hwp+zip"
├── META-INF/container.xml        ← 루트 파일 지정
├── Contents/
│   ├── content.hml               ← 본문 (핵심)
│   ├── header.xml                ← 스타일, 폰트, 문단 모양
│   └── BinData/                  ← 이미지 등 바이너리
└── Preview/
    ├── PrvImage.png              ← 미리보기 이미지
    └── PrvText.txt               ← 텍스트 미리보기
```

### .hwp (구형, 바이너리)
OLE Compound Document 포맷. `pyhwp` 라이브러리로만 파싱 가능.
직접 편집 불가 → 텍스트 추출 또는 LibreOffice로 hwpx/docx 변환 후 작업한다.

---

## 의존성 설치

```bash
# pyhwp - .hwp 바이너리 읽기용 (없으면 설치)
pip install pyhwp --break-system-packages

# pip 미설치 시
python -m pip --version || python -m ensurepip --upgrade
```

LibreOffice는 시스템에 기본 설치되어 있다 (`/usr/bin/libreoffice`).

---

## .hwp 파일 읽기

```bash
python scripts/read_hwp.py document.hwp
# → 텍스트를 stdout 또는 지정 파일로 출력
```

`read_hwp.py`는 `pyhwp`의 `hwp5txt` 모듈을 사용해 텍스트를 추출한다.
pyhwp가 설치되어 있지 않으면 LibreOffice로 docx 변환 후 텍스트를 읽는다.

---

## .hwpx 파일 언팩/팩

```bash
# 언팩: .hwpx → 디렉토리 (XML 들여쓰기 포함)
python scripts/unpack.py document.hwpx unpacked/

# 팩: 디렉토리 → .hwpx
python scripts/pack.py unpacked/ output.hwpx
```

언팩 후 `unpacked/Contents/content.hml`이 본문 XML이다.
텍스트는 `<hp:T>` 태그 안에 있다. 자세한 XML 구조는 `references/hwpx-xml.md` 참조.

---

## .hwpx 편집 (3단계 순서 준수)

### Step 1: 언팩
```bash
python scripts/unpack.py document.hwpx unpacked/
```

### Step 2: XML 편집
`unpacked/Contents/content.hml`을 직접 수정한다.
- 텍스트 수정: `<hp:T>` 태그 내용 변경
- 문단 추가: `<hp:P>` 블록 추가 (기존 문단 복사 후 수정 권장)
- **Edit 도구를 직접 사용한다** (Python 스크립트 작성 불필요)

XML 구조 상세는 `references/hwpx-xml.md`를 읽는다.

### Step 3: 팩
```bash
python scripts/pack.py unpacked/ output.hwpx --original document.hwpx
```

---

## 새 .hwpx 생성

```python
# scripts/create.py 사용법
python scripts/create.py --output new_document.hwpx

# 또는 Python에서 직접 호출
from scripts.create import HWPXDocument

doc = HWPXDocument()
doc.add_paragraph("제목", style="title")
doc.add_paragraph("본문 내용입니다.")
doc.save("output.hwpx")
```

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

- **`<hp:T>` 안에서 `\n` 사용 금지** — 새 줄은 새 `<hp:P>` 블록으로 표현
- **네임스페이스 반드시 유지** — 기존 XML의 xmlns 선언을 제거하지 않는다
- **`id` 속성 중복 금지** — 새 `<hp:P>` 추가 시 기존 최대 id + 1 사용
- **BinData 경로** — 이미지 참조 시 `Contents/BinData/` 하위 파일명과 일치해야 함

XML 전체 구조, 네임스페이스, 스타일 참조 방법은 `references/hwpx-xml.md`에서 확인한다.

---

## Dependencies

- **pyhwp**: `.hwp` 바이너리 텍스트 추출 (`pip install pyhwp --break-system-packages`)
- **LibreOffice**: 파일 변환 (`/usr/bin/libreoffice`)
- **Python stdlib**: `zipfile`, `xml.etree.ElementTree`, `lxml` (HWPX 처리)
