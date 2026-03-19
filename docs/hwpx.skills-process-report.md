# hwpx 스킬 개발 과정 보고서

**작성일**: 2026-03-19 (최종 업데이트: 2026-03-20)
**작성자**: Claude (Cowork)
**스킬 위치**: `.skills/skills/hwpx/`
**패키지 파일**: `hwpx.skill`

---

## 1. 개발 배경 및 목표

대한민국 공공기관·기업에서 광범위하게 쓰이는 **한글(HWP/HWPX) 파일**을 Cowork 에이전트가 읽고, 생성하고, 편집할 수 있도록 전용 스킬을 개발하기로 했다. 기존의 `docx` 스킬 구조를 참고하여 "docx 스킬의 한글 문서 버전"을 포팅하는 개념으로 접근했다.

**초기 목표:**
- `.hwp` (바이너리) 및 `.hwpx` (XML 기반) 파일 읽기
- 새 `.hwpx` 파일 생성
- 기존 `.hwpx` 편집 (언팩 → XML 수정 → 팩)
- HWP ↔ PDF/DOCX 변환

---

## 2. 스킬 구조 설계

`docx` 스킬을 분석하여 동일한 디렉터리 구조를 채택했다.

```
hwpx/
├── SKILL.md                 # 트리거 설명 + 워크플로 가이드
├── scripts/
│   ├── read_hwp.py          # .hwp 바이너리 텍스트 추출
│   ├── create.py            # 새 .hwpx 문서 생성
│   ├── unpack.py            # .hwpx → 디렉터리 (XML 들여쓰기)
│   ├── pack.py              # 디렉터리 → .hwpx
│   └── soffice.py           # LibreOffice 변환 래퍼
├── references/
│   └── hwpx-xml.md          # HWPX XML 구조 전체 참조
└── evals/
    └── evals.json           # 테스트 케이스
```

---

## 3. 포맷 분석: HWP vs HWPX

### 3.1 HWP (바이너리, 구형)
- OLE Compound Document 포맷
- 직접 편집 불가
- `pyhwp` 라이브러리(`hwp5`)로만 텍스트 추출 가능
- 설치: `pip install pyhwp --break-system-packages`

### 3.2 HWPX (XML 기반, 신형)
- ZIP 아카이브 내부에 XML 파일들로 구성
- KS X 6101 국가표준 기반

**초기 파악한 구조 (오류 있음 — 아래 '발견된 버그' 참조):**
```
document.hwpx (ZIP)
├── mimetype
├── META-INF/container.xml
└── Contents/content.hml     ← ❌ 실제로는 존재하지 않음
```

**실제 올바른 구조 (샘플 파일 역공학으로 확인):**
```
document.hwpx (ZIP)
├── mimetype                      ← ZIP_STORED, "application/hwp+zip"
├── version.xml
├── settings.xml
├── META-INF/container.xml        ← content.hpf 를 rootfile 로 지정
├── META-INF/manifest.xml
├── Contents/
│   ├── content.hpf               ← OPF 패키지 (header + section0 목록)
│   ├── header.xml                ← 글자모양·단락모양·스타일 정의
│   └── section0.xml              ← 본문 (<hs:sec> 루트)
└── Preview/
    ├── PrvText.txt
    └── PrvImage.png
```

---

## 4. 개발 과정

### Phase 1: .hwp 읽기 (`read_hwp.py`)

**목표:** `.hwp` 바이너리 파일에서 텍스트 추출

**시도 1 — `open_storage` 사용:**
```python
from hwp5.storage import open_storage  # ❌ ImportError
storage = open_storage(str(hwp_path))
```
→ `pyhwp 0.1b15`에 `open_storage`는 존재하지 않음

**시도 2 — subprocess로 `hwp5txt` 모듈 실행:**
```python
result = subprocess.run(["python", "-m", "hwp5.hwp5txt", str(hwp_path)],
                        capture_output=True)
```
→ exit 0이지만 stdout이 비어있음 (0 bytes). pyhwp가 `sys.stdout.buffer`(바이너리)에 직접 쓰는데, subprocess pipe가 이를 캡처하지 못함

**시도 3 (최종 해결) — 임시 파일 방식:**
```python
from hwp5.hwp5txt import TextTransform
from hwp5.xmlmodel import Hwp5File
from contextlib import closing
import tempfile

text_transform = TextTransform()
transform = text_transform.transform_hwp5_to_text

with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
    tmp_path = Path(tmp.name)

with closing(Hwp5File(str(hwp_path))) as hwp5file:
    with open(tmp_path, "wb") as dest:
        transform(hwp5file, dest)   # 임시 파일에 직접 기록

text = tmp_path.read_text(encoding="utf-8", errors="replace")
```
→ pyhwp가 바이너리 파일에 직접 쓰도록 허용하여 캡처 문제 해결

**추가 버그 — `UnboundLocalError: e1`:**
```python
# 잘못된 코드
try:
    return read_with_pyhwp(hwp_path)
except Exception as e1:
    pass
try:
    return read_with_libreoffice(hwp_path)
except Exception as e2:
    raise RuntimeError(f"pyhwp: {e1}")  # ❌ e1은 except 블록 밖에서 참조 불가

# 수정된 코드
pyhwp_err = None
try:
    return read_with_pyhwp(hwp_path)
except Exception as e:
    pyhwp_err = e
```

### Phase 2: .hwpx 언팩/팩 (`unpack.py`, `pack.py`)

- `unpack.py`: `zipfile`로 ZIP 해제 후 `lxml`(없으면 stdlib xml)로 XML 들여쓰기
- `pack.py`: 디렉터리를 다시 ZIP으로 압축. `mimetype`은 반드시 첫 번째 항목으로 `ZIP_STORED` 저장

### Phase 3: .hwpx 생성 (`create.py`) — 1차 구현 (오류 있음)

**초기 접근 — 단순 XML 직접 생성:**
```python
# ❌ 잘못된 구조
NS = 'xmlns:hml="http://www.hancom.co.kr/hwpml/2012/core" ...'  # 2012 (잘못됨)

content_hml = f"""<hml:HWPMLDocType {NS}>  <!-- ❌ 루트 요소 잘못됨 -->
  <hml:HEAD>
    <hml:CHARSHAPELIST>
      <hml:CHARSHAPE id="0" .../>   <!-- ❌ 태그 대문자 (잘못됨) -->
    </hml:CHARSHAPELIST>
  </hml:HEAD>
  <hml:BODY>
    <hml:SECTION>
      <hp:P id="0">                 <!-- ❌ 태그 대문자 (잘못됨) -->
        <hp:RUN ...>
          <hp:T>텍스트</hp:T>
        </hp:RUN>
      </hp:P>
    </hml:SECTION>
  </hml:BODY>
</hml:HWPMLDocType>"""

# 파일 구조도 잘못됨
zf.writestr("Contents/content.hml", content_hml)  # ❌ content.hml은 없는 파일
```

이 구조로 생성한 파일을 한글 앱에서 열면 **"파일이 손상되었습니다"** 오류 발생.

### Phase 4: 샘플 파일 역공학 분석

실제 HWPX 파일(`[0313]문체부보도자료-예술산업 금융지원 시범사업 실시.hwpx`)을 `unpack.py`로 분해하여 올바른 포맷을 파악했다.

**발견된 실제 포맷의 핵심 규칙:**

| 항목 | 초기 가정 (오류) | 실제 표준 |
|------|-----------------|-----------|
| 루트 파일 | `Contents/content.hml` | `Contents/content.hpf` (OPF 패키지) |
| 본문 파일 | `content.hml` | `Contents/section0.xml` |
| 스타일 파일 | `content.hml` 내부 `<hml:HEAD>` | `Contents/header.xml` (별도 파일) |
| 네임스페이스 연도 | `2012` | `2011` |
| 루트 요소 | `<hml:HWPMLDocType>` | `<hs:sec>` |
| 태그 케이스 | 대문자 `<hp:P>`, `<hp:RUN>` | 소문자 `<hp:p>`, `<hp:run>`, `<hp:t>` |
| 첫 단락 | 일반 단락 | `<hp:secPr>` 페이지 설정 필수 포함 |
| 필수 파일 | 2개 | 8개 (`version.xml`, `settings.xml` 등) |
| `container.xml` 구조 | 단순 | `ocf:container` + OPF 네임스페이스 |

### Phase 5: `create.py` 2차 구현 (올바른 포맷)

**올바른 포맷으로 전면 재작성:**

```python
# ✅ 올바른 네임스페이스 (2011)
NS_SEC = (
    'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph" '
    'xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section" '
    ...
)

# ✅ 올바른 루트 요소
section0_xml = f'<hs:sec {NS_SEC}> ... </hs:sec>'

# ✅ 올바른 태그 (소문자)
f'<hp:p id="{pid}" paraPrIDRef="{ps_id}" ...>'
f'  <hp:run charPrIDRef="{cs_id}">'
f'    <hp:t>{text}</hp:t>'
f'  </hp:run>'
f'</hp:p>'

# ✅ 첫 단락에 secPr 필수
f'<hp:run charPrIDRef="0">'
f'  <hp:secPr ...>'
f'    <hp:pagePr width="59528" height="84188" .../>'
f'  </hp:secPr>'
f'  <hp:ctrl><hp:colPr .../></hp:ctrl>'
f'</hp:run>'
```

**올바른 파일 어셈블리:**
```python
zf.writestr("mimetype", ...)           # ZIP_STORED, 첫 번째
zf.writestr("version.xml", ...)
zf.writestr("META-INF/container.xml", ...)   # → content.hpf 지정
zf.writestr("META-INF/manifest.xml", ...)
zf.writestr("Contents/content.hpf", ...)     # OPF 패키지
zf.writestr("Contents/header.xml", ...)      # 스타일 정의
zf.writestr("Contents/section0.xml", ...)    # 본문
zf.writestr("Preview/PrvText.txt", ...)
zf.writestr("settings.xml", ...)
```

> ⚠️ **2차 구현 이후에도 "파일이 손상됨" 오류가 지속되었다.** 파일 구조 자체는 맞았으나, `header.xml`의 세부 태그 구조가 표준과 달랐던 것이 원인이었다. Phase 6에서 심층 분석을 진행했다.

---

### Phase 6: `header.xml` 세부 구조 버그 심층 분석

2차 구현 이후에도 "파일이 손상됨" 오류가 지속되어, `header.xml`의 태그 구조를 실제 샘플과 세밀하게 비교하는 작업을 수행했다.

**발견된 header.xml 태그명 오류:**

| 섹션 | 초기 생성 코드 (오류) | 실제 표준 태그명 |
|------|----------------------|-----------------|
| 폰트 목록 | `<hh:fontList>` | `<hh:fontfaces>` + `<hh:fontface lang="HANGUL">` |
| 테두리 | `<hh:borderFillList>` | `<hh:borderFills>` |
| 탭 설정 | 누락 | `<hh:tabProperties>` (필수) |
| 번호 목록 | 누락 | `<hh:numberings>` (필수) |
| 스타일 목록 | `<hh:styleList>` | `<hh:styles>` |
| 메모 모양 | `<hh:memoShapeList>` | `<hh:memoProperties>` |
| 문서 호환성 | `<hh:compatible>` | `<hh:compatibleDocument>` |

**발견된 charPr 구조 오류:**

```xml
<!-- ❌ 잘못된 구조 -->
<hh:charPr id="1" bold="1" font="함초롬바탕" shadeColor="#FFFFFF">

<!-- ✅ 올바른 구조 (KS X 6101) -->
<hh:charPr id="1" height="1400" textColor="#000000" shadeColor="none" borderFillIDRef="1">
  <hh:fontRef hangul="1" latin="1" hanja="1" japanese="1" other="1" symbol="1" user="1"/>
  <hh:bold/>                         <!-- bold는 속성이 아닌 자식 요소 -->
  <hh:underline type="NONE" shape="SOLID" color="#000000"/>
  <hh:strikeout type="NONE" shape="SOLID" color="#000000"/>
  <hh:outline/>
  <hh:shadow/>
</hh:charPr>
```

**발견된 paraPr 구조 오류:**

```xml
<!-- ❌ 잘못된 구조 -->
<hh:paraPr id="0" align="JUSTIFY" textDir="LTR">

<!-- ✅ 올바른 구조 (KS X 6101) -->
<hh:paraPr id="0" tabPrIDRef="0" textDir="LTR">
  <hh:align horizontal="JUSTIFY" vertical="BASELINE"/>  <!-- align은 자식 요소 -->
  <hh:breakSetting breakLatinWord="KEEP_WORD" breakNonLatinWord="BREAK_ALL" .../>
  <hh:autoSpacing eAsianEng="0" eAsianNum="0"/>
  <hh:switch/>
  <hh:border .../>
</hh:paraPr>
```

**발견된 secPr 배치 오류:**

```xml
<!-- ❌ 잘못된 구조: 텍스트와 secPr을 같은 run에 혼합 -->
<hp:run charPrIDRef="0">
  <hp:secPr>...</hp:secPr>
  <hp:t>텍스트</hp:t>      <!-- secPr과 텍스트를 같은 run에 넣으면 안 됨 -->
</hp:run>

<!-- ✅ 올바른 구조: secPr용 run과 텍스트용 run을 분리 -->
<hp:run charPrIDRef="0"><hp:secPr>...</hp:secPr></hp:run>
<hp:run charPrIDRef="0"><hp:t>텍스트</hp:t></hp:run>
```

이러한 수정을 모두 적용했음에도 "파일이 손상됨" 오류가 지속되었다. 표준 문서 자체를 직접 확인할 필요가 있다고 판단했다.

---

### Phase 7: KS X 6101 표준 직접 확인 (Claude in Chrome 활용)

국가표준원(`standard.go.kr`)의 KS X 6101:2024 원문을 Claude in Chrome 브라우저 도구로 직접 열람했다.

**확인된 핵심 사항 — refList 순서 (section 9.2.3):**
```
fontfaces → borderFills → charProperties → tabProperties
→ numberings → bullets → paraProperties → styles → memoProperties
```

이 순서가 `header.xml`에서 반드시 지켜져야 함을 확인했다. 그러나 복잡한 XML 구조를 수동으로 정확히 재현하는 것이 근본적인 문제임이 명확해졌다.

---

### Phase 8: `pyhwpx` 라이브러리 시도 → Windows 전용으로 사용 불가

직접 XML 생성 방식의 한계를 인정하고, 기존에 검증된 한글 라이브러리를 사용하기로 방향을 전환했다.

```bash
pip install pyhwpx --break-system-packages
```

```python
import pyhwpx
# → RuntimeError: pyhwpx는 Windows에서만 동작합니다
```

`pyhwpx`는 한컴 Win32 COM 오브젝트에 의존하므로 Linux에서 사용 불가. 대안을 탐색했다.

---

### Phase 9: `python-hwpx` 라이브러리 도입 → 최종 해결

`python-hwpx` (v2.8.3) 라이브러리를 발견했다. Linux 호환 한국제 라이브러리로, `HwpxDocument.new()`가 내부적으로 KS X 6101 표준에 완전히 부합하는 ZIP 구조와 XML을 생성해준다.

**라이브러리 설치 확인:**
```bash
pip install python-hwpx --break-system-packages
# → Successfully installed python-hwpx-2.8.3
```

**핵심 API 확인:**
```python
from hwpx import HwpxDocument, TextExtractor

# 문서 생성
doc = HwpxDocument.new()
doc.add_paragraph("제목", style_id_ref=2)   # 개요 1
doc.add_paragraph("본문 내용입니다.")
doc.save_to_path("output.hwpx")

# 텍스트 읽기
te = TextExtractor("document.hwpx")
text = te.extract_text()
```

**`create.py` 완전 교체 — python-hwpx 기반 래퍼:**

직접 XML 생성 로직(수백 줄)을 모두 제거하고, `HwpxDocument.new()`를 래핑하는 `HWPXDocument` 클래스로 대체했다.

```python
from hwpx import HwpxDocument

STYLE_MAP = {
    "normal": 0, "body": 1,
    "title": 2, "h1": 2,
    "h2": 3, "h3": 4,
    ...
}

class HWPXDocument:
    def __init__(self):
        self._doc = HwpxDocument.new()

    def add_paragraph(self, text="", *, style=None, style_id=None):
        sid = STYLE_MAP.get(style.lower(), 0) if style else style_id
        self._doc.add_paragraph(text, style_id_ref=sid)
        return self

    def add_blank_line(self):
        self._doc.add_paragraph("")
        return self

    def save(self, path):
        self._doc.save_to_path(str(path))
```

**결과:** 생성된 파일이 한컴오피스(한글)에서 정상적으로 열림 ✅

> **참고:** `ensure_run_style(bold=True)` 호출 시 lxml/stdlib ElementTree 혼용 버그(`TypeError: SubElement() argument 1 must be xml.etree.ElementTree.Element, not lxml.etree._Element`)가 발생한다. 현재 bold는 개요 스타일(`style="title"` 등)로 시각적 강조를 대체하며, 이 라이브러리 버그는 upstream 이슈로 남겨둔다.

---

## 5. 발견된 버그 목록 및 수정 내역

| # | 버그 | 원인 | 수정 |
|---|------|------|------|
| 1 | `ImportError: open_storage` | pyhwp 0.1b15에 존재하지 않는 API | `Hwp5File` + `TextTransform` 직접 사용 |
| 2 | subprocess stdout 비어있음 | pyhwp가 `sys.stdout.buffer`에 쓰지만 pipe 캡처 안 됨 | 임시 파일에 직접 기록 후 읽기 |
| 3 | `UnboundLocalError: e1` | except 블록 변수를 외부에서 참조 | `pyhwp_err = None` 사전 선언으로 변경 |
| 4 | "파일이 손상됨" 1차 | 잘못된 파일 구조 · 2012 네임스페이스 · 대문자 태그 · content.hml | 샘플 역공학으로 올바른 포맷 파악 후 재작성 |
| 5 | `<hp:T>` 내 개행 문자 | 다중 줄 텍스트를 `\n`으로 처리 | 각 줄을 별도 `<hp:p>` 블록으로 분리 |
| 6 | 패키지 재빌드 미반영 | 버그 수정 후 `hwpx.skill`을 재패키징하지 않음 | 수정 직후 즉시 재패키징하도록 절차 정립 |
| 7 | "파일이 손상됨" 2차 — `header.xml` 태그명 | `<hh:fontList>`, `<hh:styleList>` 등 비표준 태그명 사용 | 실제 샘플과 비교해 `fontfaces`, `styles` 등 수정 |
| 8 | "파일이 손상됨" 3차 — `charPr` 구조 | `bold` 속성 → `<hh:bold/>` 자식 요소, `<hh:font>` → `<hh:fontRef>` | KS X 6101 표준 및 샘플 분석으로 수정 |
| 9 | "파일이 손상됨" 4차 — `paraPr` 구조 | `align` 속성 → `<hh:align>` 자식 요소, 필수 자식 요소 누락 | 표준 확인 후 수정 |
| 10 | "파일이 손상됨" 5차 — `secPr` 배치 | `secPr`과 텍스트를 같은 `<hp:run>`에 혼합 | secPr run과 텍스트 run을 별도 분리 |
| 11 | `pyhwpx` Windows 전용 | Win32 COM 의존 — Linux에서 `RuntimeError` | `python-hwpx`(Linux 호환)로 대체 |
| 12 | `ensure_run_style(bold=True)` TypeError | `python-hwpx` 내부에서 lxml과 stdlib ET 혼용 | bold 미사용, 제목은 개요 스타일로 대체 (upstream 버그) |

---

## 6. 실제 문서 처리 테스트 결과

### 테스트 1: `.hwp` 읽기 (문화특화지역 조성사업 현황)

**파일:** `문화특화지역(문화도시문화마을) 조성 사업 현황.hwp`
**결과:** `read_hwp.py` (임시 파일 방식) 으로 27줄 텍스트 추출 성공
**추출 내용:** 사업개요·지원대상·추진체계·'15년 지원현황

### 테스트 2: `.hwpx` 읽기 (예술산업 금융지원 보도자료)

**파일:** `[0313]문체부보도자료-예술산업 금융지원 시범사업 실시.hwpx`
**결과:** `unpack.py` + XML 파싱으로 전문 추출 성공
**추출 내용:** 융자 200억·보증 237.5억·신청 일정·담당 연락처 등 전체 내용

### 테스트 3: `.hwpx` 생성 시도 — 반복 실패 (Phase 3~7)

**파일:** 새로 생성 시도
**결과:** 1차(잘못된 포맷) → 2차(파일 구조 수정) → 3차(header.xml 태그명) → 4차(charPr 구조) → 5차(paraPr 구조) → 6차(secPr 분리) 모두 "파일이 손상됨" 지속

직접 XML 생성 방식으로는 KS X 6101 표준의 복잡한 header.xml 구조를 완전히 재현하기 어렵다는 결론에 도달. python-hwpx 라이브러리로 전환을 결정.

### 테스트 4: `.hwpx` 생성 (python-hwpx 적용) — 성공

**파일:** `예술산업_금융지원_브로셔_요약.hwpx`
**방법:** `TextExtractor`로 브로셔 원문 추출 → 요약 작성 → `HWPXDocument`(python-hwpx 래퍼)로 저장
**결과:** 한컴오피스(한글)에서 정상적으로 열림 ✅ (8,738 bytes)
**내용:** 사업 개요, 융자(200억)/보증(237.5억) 상세, 신청 방법, 문의처, 주의사항

---

## 7. HWPX XML 핵심 규칙 (개발 중 확립)

```
1. 네임스페이스: 2011 (http://www.hancom.co.kr/hwpml/2011/*)
2. 태그명: 소문자 (hp:p, hp:run, hp:t, hs:sec, hh:charPr ...)
3. 파일 구조: content.hpf → header.xml + section0.xml
4. mimetype: ZIP_STORED, ZIP 내 첫 번째 항목
5. 첫 <hp:p>의 첫 <hp:run>에 반드시 <hp:secPr> 포함
6. <hp:t> 안에서 줄바꿈(\n) 금지 → 새 <hp:p> 블록으로 표현
7. 단락 id: 문서 전체에서 유일한 정수
8. charPr id는 header.xml의 <hh:charPr> id 와 일치해야 함
```

---

## 8. 최종 스킬 파일 목록

| 파일 | 역할 | 상태 |
|------|------|------|
| `SKILL.md` | 트리거 조건 + 퀵 레퍼런스 + 워크플로 | ✅ python-hwpx 반영 업데이트 완료 |
| `scripts/read_hwp.py` | .hwp (pyhwp) · .hwpx (python-hwpx TextExtractor) 텍스트 추출 | ✅ |
| `scripts/create.py` | 새 .hwpx 생성 — **python-hwpx `HwpxDocument.new()` 기반** | ✅ 완전 교체 (Phase 9) |
| `scripts/unpack.py` | .hwpx → 디렉터리 (XML 들여쓰기) | ✅ |
| `scripts/pack.py` | 디렉터리 → .hwpx (mimetype 첫 번째) | ✅ |
| `scripts/soffice.py` | LibreOffice 변환 래퍼 | ✅ |
| `references/hwpx-xml.md` | HWPX XML 구조 전체 참조 | ✅ |
| `evals/evals.json` | 테스트 케이스 | ✅ |

---

## 9. 알려진 한계 및 향후 개선 사항

| 항목 | 현황 | 개선 방향 |
|------|------|-----------|
| Bold 텍스트 | `python-hwpx` 내부 버그로 `ensure_run_style(bold=True)` 불가 | upstream 버그 수정 대기; 임시로 제목 스타일로 대체 |
| 표(TABLE) 생성 | `create.py`에 미구현 | `python-hwpx`의 `add_table()` API 활용 검토 |
| 이미지 삽입 | 미구현 | `python-hwpx`의 `add_image()` API 활용 검토 |
| LibreOffice HWPX 변환 | HWPX → PDF 변환 실패 (LibreOffice HWPX 지원 미흡) | HWP(바이너리) 변환 또는 한컴 뷰어 활용 검토 |
| `.hwp` 쓰기 | 불가 (바이너리 포맷) | HWPX로 생성 후 한컴오피스에서 변환 안내 |
| .hwpx 직접 편집 시 XML 구조 | unpack→편집→pack 방식으로만 가능 | `python-hwpx`의 XML API를 통한 프로그래밍 편집 검토 |

---

## 10. 의존성

```bash
# .hwpx 생성·읽기용 (핵심, Linux 호환)
pip install python-hwpx --break-system-packages

# .hwp 바이너리 읽기용
pip install pyhwp --break-system-packages

# LibreOffice (시스템 기본 설치)
/usr/bin/libreoffice

# Python 표준 라이브러리 (unpack/pack용)
zipfile, xml.etree.ElementTree, xml.sax.saxutils
```

> **주의:** `pyhwpx`는 Windows 전용 (Win32 COM 의존). Linux에서는 `python-hwpx`를 사용한다.

---

## 11. 개발 이력 타임라인

| 날짜 | 작업 | 결과 |
|------|------|------|
| 2026-03-19 | 스킬 구조 설계, read_hwp.py 개발 | `.hwp` 읽기 성공 |
| 2026-03-19 | create.py 1차 — 직접 XML 생성 (content.hml 방식) | "파일이 손상됨" |
| 2026-03-19 | 샘플 역공학 분석, create.py 2차 — 올바른 파일 구조 | "파일이 손상됨" 지속 |
| 2026-03-19 | header.xml 태그명 수정 (fontfaces, borderFills 등) | "파일이 손상됨" 지속 |
| 2026-03-19 | charPr·paraPr 구조 수정, secPr run 분리 | "파일이 손상됨" 지속 |
| 2026-03-20 | KS X 6101 표준 직접 확인 (Claude in Chrome) | 표준 확인 완료 |
| 2026-03-20 | pyhwpx 시도 → Windows 전용으로 불가 | 대안 탐색 |
| 2026-03-20 | python-hwpx 도입, create.py 완전 교체 | ✅ 정상 열림 |
| 2026-03-20 | 브로셔 요약 HWPX 생성 (python-hwpx 기반) | ✅ 성공 |
| 2026-03-20 | SKILL.md 업데이트, hwpx.skill 재패키징 | ✅ 완료 |
