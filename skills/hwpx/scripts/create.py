#!/usr/bin/env python3
"""
create.py — 새 .hwpx 문서를 생성한다.

KS X 6101 규격 준수:
  - 2011 네임스페이스 (http://www.hancom.co.kr/hwpml/2011/*)
  - 소문자 태그 (<hp:p>, <hp:run>, <hp:t>)
  - section0.xml 이 본문 파일
  - content.hpf 가 OPF 패키지 파일
  - 첫 번째 <hp:p> 의 <hp:run> 안에 <hp:secPr> 포함 (필수)
  - 총 9개 파일로 구성된 ZIP 아카이브

사용법 (CLI):
    python scripts/create.py --output new_doc.hwpx
    python scripts/create.py --output report.hwpx --title "보고서 제목"

사용법 (Python API):
    from scripts.create import HWPXDocument
    doc = HWPXDocument()
    doc.add_paragraph("제목", style="title")
    doc.add_paragraph("소제목", style="heading")
    doc.add_paragraph("본문 내용입니다.")
    doc.add_paragraph("")  # 빈 줄
    doc.save("output.hwpx")
"""

import zipfile
import argparse
from pathlib import Path
from xml.sax.saxutils import escape


# ── 네임스페이스 ──────────────────────────────────────────────────────
NS_SEC = (
    'xmlns:ha="http://www.hancom.co.kr/hwpml/2011/app" '
    'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph" '
    'xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section" '
    'xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core" '
    'xmlns:hh="http://www.hancom.co.kr/hwpml/2011/head" '
    'xmlns:hpf="http://www.hancom.co.kr/schema/2011/hpf"'
)

NS_HEAD = 'xmlns:hh="http://www.hancom.co.kr/hwpml/2011/head"'

# ── 스타일 정의 ───────────────────────────────────────────────────────
# height 단위: 1/100 pt  (1000 = 10pt, 1400 = 14pt, 2000 = 20pt)
CHAR_STYLES = {
    "body":    {"id": 0, "height": 1000, "bold": "0"},
    "heading": {"id": 1, "height": 1400, "bold": "1"},
    "title":   {"id": 2, "height": 2000, "bold": "1"},
}

PARA_STYLES = {
    "body":    {"id": 0, "align": "JUSTIFY", "above": "0",   "below": "0"},
    "heading": {"id": 1, "align": "LEFT",    "above": "300", "below": "100"},
    "title":   {"id": 2, "align": "CENTER",  "above": "0",   "below": "500"},
}


# ── header.xml 생성 ───────────────────────────────────────────────────

def _charPr_xml(cid: int, height: int, bold: str) -> str:
    """charPr 엔트리 1개 생성."""
    font_tags = (
        '<hh:font hangul="4" latin="4" hanja="4" japanese="4" other="4" symbol="4" user="4"/>'
        '<hh:ratio hangul="100" latin="100" hanja="100" japanese="100"'
        ' other="100" symbol="100" user="100"/>'
        '<hh:spacing hangul="0" latin="0" hanja="0" japanese="0"'
        ' other="0" symbol="0" user="0"/>'
        '<hh:relSz hangul="100" latin="100" hanja="100" japanese="100"'
        ' other="100" symbol="100" user="100"/>'
        '<hh:offset hangul="0" latin="0" hanja="0" japanese="0"'
        ' other="0" symbol="0" user="0"/>'
    )
    return (
        f'<hh:charPr id="{cid}" height="{height}" textColor="#000000"'
        f' shadeColor="#FFFFFF" useFontSpace="0" useKerning="0"'
        f' symMark="NONE" borderFillIDRef="0"'
        f' bold="{bold}" italic="0" underline="0" strikeout="0"'
        f' outline="0" shadow="0" emboss="0" engrave="0"'
        f' superScript="0" subScript="0" smallCaps="0" allCaps="0"'
        f' hiddenText="0" lang="0" spacing="0" ratio="100" tag="">'
        f'{font_tags}</hh:charPr>'
    )


def _build_header_xml() -> str:
    """최소한의 유효한 header.xml을 생성 (샘플 파일 불필요)."""
    charpr_list = "\n    ".join(
        _charPr_xml(cs["id"], cs["height"], cs["bold"])
        for cs in CHAR_STYLES.values()
    )
    parapr_list = "\n    ".join(
        f'<hh:paraPr id="{ps["id"]}" lineSpacing="160" lineSpacingType="PERCENT"'
        f' spaceAbove="{ps["above"]}" spaceBelow="{ps["below"]}"'
        f' align="{ps["align"]}" tabStop="8000" outline="0" break="SECTION"'
        f' vertAlign="BASELINE" widowOrphan="0" keepWithNext="0" keepLines="0"'
        f' pageBreakBefore="0" fontLineHeight="0" snapToGrid="1"'
        f' condense="0" mergedParaNumber="0" checked="0"/>'
        for ps in PARA_STYLES.values()
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes" ?><hh:head {NS_HEAD}>
  <hh:beginNum page="1" footnote="1" endnote="1" pic="1" tbl="1" equation="1"/>
  <hh:refList>
    <hh:fontList>
      <hh:font id="0" name="함초롬바탕" type="UNKNOWN_FONT"/>
      <hh:font id="1" name="함초롬돋움" type="UNKNOWN_FONT"/>
      <hh:font id="2" name="Arial" type="UNKNOWN_FONT"/>
      <hh:font id="3" name="Times New Roman" type="UNKNOWN_FONT"/>
      <hh:font id="4" name="맑은 고딕" type="UNKNOWN_FONT"/>
    </hh:fontList>
    <hh:borderFillList>
      <hh:borderFill id="0" threeD="0" shadow="0" centerLine="0" breakCellSeparateLine="0">
        <hh:slash type="NONE" crooked="0" isCounter="0"/>
        <hh:backSlash type="NONE" crooked="0" isCounter="0"/>
        <hh:leftBorder type="NONE" width="0.1 mm" color="#000000"/>
        <hh:rightBorder type="NONE" width="0.1 mm" color="#000000"/>
        <hh:topBorder type="NONE" width="0.1 mm" color="#000000"/>
        <hh:bottomBorder type="NONE" width="0.1 mm" color="#000000"/>
        <hh:diagonal type="NONE" width="0.1 mm" color="#000000"/>
        <hh:fillInfo><hh:noFill/></hh:fillInfo>
      </hh:borderFill>
      <hh:borderFill id="1" threeD="0" shadow="0" centerLine="0" breakCellSeparateLine="0">
        <hh:slash type="NONE" crooked="0" isCounter="0"/>
        <hh:backSlash type="NONE" crooked="0" isCounter="0"/>
        <hh:leftBorder type="NONE" width="0.1 mm" color="#000000"/>
        <hh:rightBorder type="NONE" width="0.1 mm" color="#000000"/>
        <hh:topBorder type="NONE" width="0.1 mm" color="#000000"/>
        <hh:bottomBorder type="NONE" width="0.1 mm" color="#000000"/>
        <hh:diagonal type="NONE" width="0.1 mm" color="#000000"/>
        <hh:fillInfo><hh:noFill/></hh:fillInfo>
      </hh:borderFill>
    </hh:borderFillList>
    <hh:charProperties>
    {charpr_list}
    </hh:charProperties>
    <hh:paraProperties>
    {parapr_list}
    </hh:paraProperties>
    <hh:styleList>
      <hh:style id="0" name="바탕글" engName="Normal" type="PARA"
                nextStyleIDRef="0" charPrIDRef="0" paraPrIDRef="0" lockForm="0"/>
    </hh:styleList>
    <hh:memoShapeList>
      <hh:memoShape id="1"/>
    </hh:memoShapeList>
    <hh:trackChangeList/>
  </hh:refList>
  <hh:compatible>
    <hh:layoutCompatibility>
      <hh:splitPageContents/>
    </hh:layoutCompatibility>
  </hh:compatible>
</hh:head>"""


# ── secPr (섹션 속성, A4 세로) ────────────────────────────────────────

def _secPr_xml() -> str:
    """A4 세로 기본 여백 섹션 속성 XML."""
    return (
        '<hp:secPr id="" textDirection="HORIZONTAL" spaceColumns="1134"'
        ' tabStop="8000" tabStopVal="4000" tabStopUnit="HWPUNIT"'
        ' outlineShapeIDRef="1" memoShapeIDRef="1"'
        ' textVerticalWidthHead="0" masterPageCnt="0">'
        '<hp:grid lineGrid="0" charGrid="0" wonggojiFormat="0"/>'
        '<hp:startNum pageStartsOn="BOTH" page="0" pic="0" tbl="0" equation="0"/>'
        '<hp:visibility hideFirstHeader="0" hideFirstFooter="0"'
        ' hideFirstMasterPage="0" border="SHOW_ALL" fill="SHOW_ALL"'
        ' hideFirstPageNum="0" hideFirstEmptyLine="0" showLineNumber="0"/>'
        '<hp:lineNumberShape restartType="0" countBy="0" distance="0" startNumber="0"/>'
        '<hp:pagePr landscape="PORTRAIT" width="59528" height="84188" gutterType="LEFT_ONLY">'
        '<hp:margin header="4252" footer="4252" gutter="0"'
        ' left="8504" right="8504" top="5669" bottom="4252"/>'
        '</hp:pagePr>'
        '<hp:footNotePr>'
        '<hp:autoNumFormat type="DIGIT" userChar="" prefixChar="" suffixChar=")" supscript="0"/>'
        '<hp:noteLine length="-1" type="SOLID" width="0.1 mm" color="#000000"/>'
        '<hp:noteSpacing betweenNotes="850" belowLine="567" aboveLine="567"/>'
        '<hp:numbering type="CONTINUOUS" newNum="1"/>'
        '<hp:placement place="EACH_COLUMN" beneathText="0"/>'
        '</hp:footNotePr>'
        '<hp:endNotePr>'
        '<hp:autoNumFormat type="DIGIT" userChar="" prefixChar="" suffixChar=")" supscript="0"/>'
        '<hp:noteLine length="-1" type="SOLID" width="0.1 mm" color="#000000"/>'
        '<hp:noteSpacing betweenNotes="850" belowLine="567" aboveLine="567"/>'
        '<hp:numbering type="CONTINUOUS" newNum="1"/>'
        '<hp:placement place="END_OF_DOCUMENT" beneathText="0"/>'
        '</hp:endNotePr>'
        '<hp:pageBorderFill type="BOTH" borderFillIDRef="1" textBorder="PAPER"'
        ' headerInside="0" footerInside="0" fillArea="PAPER">'
        '<hp:offset left="1417" right="1417" top="1417" bottom="1417"/>'
        '</hp:pageBorderFill>'
        '<hp:pageBorderFill type="EVEN" borderFillIDRef="1" textBorder="PAPER"'
        ' headerInside="0" footerInside="0" fillArea="PAPER">'
        '<hp:offset left="1417" right="1417" top="1417" bottom="1417"/>'
        '</hp:pageBorderFill>'
        '<hp:pageBorderFill type="ODD" borderFillIDRef="1" textBorder="PAPER"'
        ' headerInside="0" footerInside="0" fillArea="PAPER">'
        '<hp:offset left="1417" right="1417" top="1417" bottom="1417"/>'
        '</hp:pageBorderFill>'
        '</hp:secPr>'
        '<hp:ctrl>'
        '<hp:colPr id="" type="NEWSPAPER" layout="LEFT"'
        ' colCount="1" sameSz="1" sameGap="0"/>'
        '</hp:ctrl>'
    )


# ── HWPXDocument ─────────────────────────────────────────────────────

class HWPXDocument:
    """
    간단한 HWPX 문서 생성기.

    KS X 6101 / 2011 네임스페이스 준수.
    샘플 파일 없이 독립적으로 동작한다.
    """

    def __init__(self):
        self._paragraphs: list[tuple[str, str]] = []   # (text, style)
        self._pid = 0

    def add_paragraph(self, text: str, style: str = "body"):
        """
        단락 추가.

        Args:
            text:  단락 텍스트 (빈 문자열이면 빈 줄)
            style: "title" | "heading" | "body" (기본값)
        """
        if style not in CHAR_STYLES:
            style = "body"
        self._paragraphs.append((text, style))

    def _next_pid(self) -> int:
        v = self._pid
        self._pid += 1
        return v

    def _para_xml(self, text: str, style: str, is_first: bool = False) -> str:
        """단락 하나의 XML 생성.

        is_first=True 이면 <hp:secPr>을 포함한다 (HWPX 필수 규칙).
        """
        cs = CHAR_STYLES[style]
        ps = PARA_STYLES[style]
        pid = self._next_pid()
        secpr = _secPr_xml() if is_first else ""

        if text:
            run_xml = (
                f'<hp:run charPrIDRef="{cs["id"]}">'
                f'{secpr}'
                f'<hp:t>{escape(text)}</hp:t>'
                f'</hp:run>'
            )
        else:
            # 빈 줄
            if is_first:
                run_xml = f'<hp:run charPrIDRef="0">{secpr}</hp:run>'
            else:
                run_xml = ""

        return (
            f'<hp:p id="{pid}" paraPrIDRef="{ps["id"]}" styleIDRef="0"'
            f' pageBreak="0" columnBreak="0" merged="0">'
            f'{run_xml}</hp:p>'
        )

    def _build_section0(self) -> str:
        """section0.xml 생성."""
        if self._paragraphs:
            paras = [
                self._para_xml(text, style, is_first=(i == 0))
                for i, (text, style) in enumerate(self._paragraphs)
            ]
        else:
            # 단락이 없어도 secPr을 가진 빈 단락 최소 1개 필요
            pid = self._next_pid()
            paras = [
                f'<hp:p id="{pid}" paraPrIDRef="0" styleIDRef="0"'
                f' pageBreak="0" columnBreak="0" merged="0">'
                f'<hp:run charPrIDRef="0">{_secPr_xml()}</hp:run></hp:p>'
            ]

        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>'
            f'<hs:sec {NS_SEC}>\n  '
            + "\n  ".join(paras)
            + "\n</hs:sec>"
        )

    def _build_content_hpf(self, title: str = "") -> str:
        """content.hpf (OPF 패키지) 생성."""
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>'
            '<opf:package'
            ' xmlns:ha="http://www.hancom.co.kr/hwpml/2011/app"'
            ' xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"'
            ' xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"'
            ' xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core"'
            ' xmlns:hh="http://www.hancom.co.kr/hwpml/2011/head"'
            ' xmlns:hpf="http://www.hancom.co.kr/schema/2011/hpf"'
            ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
            ' xmlns:opf="http://www.idpf.org/2007/opf/"'
            ' version="" unique-identifier="" id="">'
            '<opf:metadata>'
            f'<opf:title>{escape(title)}</opf:title>'
            '<opf:language>ko</opf:language>'
            '<opf:meta name="creator" content="text"/>'
            '</opf:metadata>'
            '<opf:manifest>'
            '<opf:item id="header" href="Contents/header.xml" media-type="application/xml"/>'
            '<opf:item id="section0" href="Contents/section0.xml" media-type="application/xml"/>'
            '<opf:item id="settings" href="settings.xml" media-type="application/xml"/>'
            '</opf:manifest>'
            '<opf:spine>'
            '<opf:itemref idref="header" linear="yes"/>'
            '<opf:itemref idref="section0" linear="yes"/>'
            '</opf:spine>'
            '</opf:package>'
        )

    def save(self, output_path: str, title: str = ""):
        """문서를 .hwpx 파일로 저장.

        생성 파일 구조 (9개):
          mimetype               ZIP_STORED, 첫 번째
          version.xml
          META-INF/container.xml
          META-INF/manifest.xml
          Contents/content.hpf   OPF 패키지
          Contents/header.xml    스타일·폰트 정의
          Contents/section0.xml  본문
          Preview/PrvText.txt
          settings.xml
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ── 각 파일 내용 빌드 ──────────────────────────────────────────
        section0 = self._build_section0()
        header   = _build_header_xml()
        hpf      = self._build_content_hpf(title)

        container = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>'
            '<ocf:container'
            ' xmlns:ocf="urn:oasis:names:tc:opendocument:xmlns:container"'
            ' xmlns:hpf="http://www.hancom.co.kr/schema/2011/hpf">'
            '<ocf:rootfiles>'
            '<ocf:rootfile full-path="Contents/content.hpf"'
            ' media-type="application/hwpml-package+xml"/>'
            '<ocf:rootfile full-path="Preview/PrvText.txt"'
            ' media-type="text/plain"/>'
            '</ocf:rootfiles>'
            '</ocf:container>'
        )
        manifest = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>'
            '<odf:manifest'
            ' xmlns:odf="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"/>'
        )
        version = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>'
            '<ha:version xmlns:ha="http://www.hancom.co.kr/hwpml/2011/app"'
            ' appVersion="9.0.0.0" fileVersion="1.3.0.0"'
            ' ooxmlCompatVersion="OOXML_2007"/>'
        )
        settings = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>'
            '<ha:docsettings xmlns:ha="http://www.hancom.co.kr/hwpml/2011/app"'
            ' spellCheckIgnoreCapital="0" spellCheckIgnoreAllCaps="0"'
            ' spellCheckIgnoreAutoHyphen="0" spellCheckIgnoreInternetAddress="0"'
            ' spellCheckIgnoreNumeric="0" spellCheckIgnoreDoubleSpaces="0"'
            ' trackChanges="0" showChanges="0" protectPassword=""'
            ' encryptVersion="0" securityLevel="0" printProtect="0"'
            ' editProtect="0" copyProtect="0" formProtect="0"'
            ' annotationPrint="0" annotationShowAuthor="1"/>'
        )
        prv_text = "\n".join(t for t, _ in self._paragraphs if t)[:200]

        # ── ZIP 아카이브 작성 ──────────────────────────────────────────
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # mimetype: 압축 없이 첫 번째 (HWPX 표준 요구 사항)
            mi = zipfile.ZipInfo("mimetype")
            mi.compress_type = zipfile.ZIP_STORED
            zf.writestr(mi, "application/hwp+zip")

            zf.writestr("version.xml",              version)
            zf.writestr("META-INF/container.xml",   container)
            zf.writestr("META-INF/manifest.xml",    manifest)
            zf.writestr("Contents/content.hpf",     hpf)
            zf.writestr("Contents/header.xml",      header)
            zf.writestr("Contents/section0.xml",    section0)
            zf.writestr("Preview/PrvText.txt",      prv_text)
            zf.writestr("settings.xml",             settings)

        print(f"✅ 문서 생성 완료: {output_path}")
        print(f"   단락 수: {len(self._paragraphs)}")


def main():
    parser = argparse.ArgumentParser(description="새 .hwpx 문서 생성")
    parser.add_argument("--output", "-o", default="new_document.hwpx",
                        help="출력 파일 경로 (기본: new_document.hwpx)")
    parser.add_argument("--title", default="",
                        help="문서 제목 (선택)")
    args = parser.parse_args()

    doc = HWPXDocument()
    if args.title:
        doc.add_paragraph(args.title, style="title")
        doc.add_paragraph("")
    doc.add_paragraph("내용을 여기에 입력하세요.")
    doc.save(args.output, title=args.title)


if __name__ == "__main__":
    main()
