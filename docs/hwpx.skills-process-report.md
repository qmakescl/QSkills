# hwpx 스킬 개발 과정 보고서

**작성일**: 2026-03-19
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

---

## 5. 발견된 버그 목록 및 수정 내역

| # | 버그 | 원인 | 수정 |
|---|------|------|------|
| 1 | `ImportError: open_storage` | pyhwp 0.1b15에 존재하지 않는 API | `Hwp5File` + `TextTransform` 직접 사용 |
| 2 | subprocess stdout 비어있음 | pyhwp가 `sys.stdout.buffer`에 쓰지만 pipe 캡처 안 됨 | 임시 파일에 직접 기록 후 읽기 |
| 3 | `UnboundLocalError: e1` | except 블록 변수를 외부에서 참조 | `pyhwp_err = None` 사전 선언으로 변경 |
| 4 | "파일이 손상됨" (HWPX) | 잘못된 파일 구조 · 네임스페이스 · 태그명 | 샘플 역공학으로 올바른 포맷 파악 후 전면 재작성 |
| 5 | `<hp:T>` 내 개행 문자 | 다중 줄 텍스트를 `\n`으로 처리 | 각 줄을 별도 `<hp:p>` 블록으로 분리 |
| 6 | 패키지 재빌드 미반영 | 버그 수정 후 `hwpx.skill`을 재패키징하지 않음 | `read_hwp.py` 수정 후 즉시 재패키징 실행 |

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

### 테스트 3: `.hwpx` 생성 (예술산업 금융지원 브로셔)

**결과:** 1차 생성 → "파일이 손상됨" 오류 → 포맷 분석 → 2차 재작성 → 한글 앱에서 정상 열림
**생성 파일:** `예술산업_금융지원_브로셔.hwpx`

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
| `SKILL.md` | 트리거 조건 + 퀵 레퍼런스 + 워크플로 | ✅ |
| `scripts/read_hwp.py` | .hwp 텍스트 추출 (pyhwp API + LibreOffice 폴백) | ✅ 버그 수정 완료 |
| `scripts/create.py` | 새 .hwpx 생성 (올바른 2011 포맷) | ✅ 전면 재작성 완료 |
| `scripts/unpack.py` | .hwpx → 디렉터리 (XML 들여쓰기) | ✅ |
| `scripts/pack.py` | 디렉터리 → .hwpx (mimetype 첫 번째) | ✅ |
| `scripts/soffice.py` | LibreOffice 변환 래퍼 | ✅ |
| `references/hwpx-xml.md` | HWPX XML 구조 전체 참조 | ✅ |
| `evals/evals.json` | 테스트 케이스 | ✅ |

---

## 9. 알려진 한계 및 향후 개선 사항

| 항목 | 현황 | 개선 방향 |
|------|------|-----------|
| 표(TABLE) 생성 | `create.py`에 미구현 | `<hp:tbl>` + `<hp:tr>` + `<hp:tc>` + `<hp:subList>` 구조 추가 |
| 이미지 삽입 | 미구현 | `BinData/` 폴더 + `<hp:pic>` 요소 추가 |
| LibreOffice HWPX 변환 | HWPX → PDF 변환 실패 (LibreOffice HWPX 지원 미흡) | HWP(바이너리) 변환 또는 한컴 뷰어 활용 검토 |
| `create.py` 스타일 | 5개 charPr·5개 paraPr만 지원 | 더 다양한 스타일(색상, 밑줄 등) 확장 가능 |
| `.hwp` 쓰기 | 불가 (바이너리 포맷) | HWPX로 생성 후 한컴오피스에서 변환 안내 |

---

## 10. 의존성

```bash
# .hwp 읽기용 (필수)
pip install pyhwp --break-system-packages

# LibreOffice (시스템 기본 설치)
/usr/bin/libreoffice

# Python 표준 라이브러리
zipfile, xml.etree.ElementTree, xml.sax.saxutils
```
