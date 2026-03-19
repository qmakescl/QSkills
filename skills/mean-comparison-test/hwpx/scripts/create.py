#!/usr/bin/env python3
"""
create.py — 새 .hwpx 문서를 생성한다.

HWPX에서 글자 서식(크기, 굵기)은 HEAD의 CHARSHAPE에 정의하고,
본문 RUN의 charPrIDRef 속성으로 참조한다.

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


# HWPX 핵심 네임스페이스
NS_DECLS = (
    'xmlns:hml="http://www.hancom.co.kr/hwpml/2012/core" '
    'xmlns:hp="http://www.hancom.co.kr/hwpml/2012/paragraph" '
    'xmlns:hs="http://www.hancom.co.kr/hwpml/2012/section" '
    'xmlns:ha="http://www.hancom.co.kr/hwpml/2012/app" '
    'xmlns:hh="http://www.hancom.co.kr/hwpml/2012/head" '
    'xmlns:hf="http://www.hancom.co.kr/hwpml/2012/fill"'
)

# 스타일별 CHARSHAPE 정의 (id → 속성)
# height 단위: 1/100 pt (1000 = 10pt)
CHARSHAPES = {
    "body":    {"id": 0, "height": 1000, "bold": "0"},  # 본문 10pt
    "heading": {"id": 1, "height": 1400, "bold": "1"},  # 소제목 14pt 굵게
    "title":   {"id": 2, "height": 2000, "bold": "1"},  # 제목 20pt 굵게
}

# 스타일별 PARASHAPE 정의 (id → 속성)
PARASHAPES = {
    "body":    {"id": 0, "align": "justify", "spaceAbove": "0",  "spaceBelow": "0"},
    "heading": {"id": 1, "align": "left",    "spaceAbove": "300","spaceBelow": "100"},
    "title":   {"id": 2, "align": "center",  "spaceAbove": "0",  "spaceBelow": "500"},
}


class HWPXDocument:
    """간단한 HWPX 문서 생성기."""

    def __init__(self):
        self._paragraphs = []   # (text, style)

    def add_paragraph(self, text: str, style: str = "body"):
        """
        단락 추가.

        Args:
            text:  단락 텍스트 (빈 문자열이면 빈 줄)
            style: "title" | "heading" | "body" (기본값)
        """
        if style not in CHARSHAPES:
            style = "body"
        self._paragraphs.append((text, style))

    def _build_charshape_xml(self) -> str:
        items = []
        for style, cs in CHARSHAPES.items():
            items.append(
                f'<hml:CHARSHAPE id="{cs["id"]}" height="{cs["height"]}" '
                f'bold="{cs["bold"]}" italic="0" underline="0" strikeout="0">'
                f'<hml:FONTID face="함초롬바탕" lang="ko"/>'
                f'<hml:FONTID face="Arial" lang="en"/>'
                f'</hml:CHARSHAPE>'
            )
        return "\n      ".join(items)

    def _build_parashape_xml(self) -> str:
        items = []
        for style, ps in PARASHAPES.items():
            items.append(
                f'<hml:PARASHAPE id="{ps["id"]}" '
                f'lineSpacing="160" lineSpacingType="percent" '
                f'spaceAbove="{ps["spaceAbove"]}" spaceBelow="{ps["spaceBelow"]}" '
                f'align="{ps["align"]}"/>'
            )
        return "\n      ".join(items)

    def _para_xml(self, text: str, style: str, pid: int) -> str:
        """단락 하나의 XML 생성."""
        cs = CHARSHAPES[style]
        ps = PARASHAPES[style]

        if text:
            escaped = escape(text)
            run_xml = (
                f'<hp:RUN charPrIDRef="{cs["id"]}">'
                f'<hp:T>{escaped}</hp:T>'
                f'</hp:RUN>'
            )
        else:
            run_xml = ""

        return (
            f'<hp:P id="{pid}" paraPrIDRef="{ps["id"]}" styleIDRef="0">'
            f'{run_xml}'
            f'</hp:P>'
        )

    def _build_content_hml(self) -> str:
        """content.hml 전체 XML 문자열 생성."""
        paras_xml = "\n      ".join(
            self._para_xml(text, style, i)
            for i, (text, style) in enumerate(self._paragraphs)
        )

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<hml:HWPMLDocType {NS_DECLS} version="1.3.0.0">
  <hml:HEAD>
    <hml:DOCPROPERTY>
      <hml:TITLE/>
    </hml:DOCPROPERTY>
    <hml:CHARSHAPELIST>
      {self._build_charshape_xml()}
    </hml:CHARSHAPELIST>
    <hml:PARASHAPELIST>
      {self._build_parashape_xml()}
    </hml:PARASHAPELIST>
    <hml:STYLELIST>
      <hml:STYLE id="0" name="바탕글" charPrIDRef="0" paraPrIDRef="0" type="para"/>
    </hml:STYLELIST>
  </hml:HEAD>
  <hml:BODY>
    <hml:SECTION>
      {paras_xml}
    </hml:SECTION>
  </hml:BODY>
</hml:HWPMLDocType>
"""

    def _build_container_xml(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="Contents/content.hml"
              media-type="application/hwp+zip"/>
  </rootfiles>
</container>
"""

    def save(self, output_path: str):
        """문서를 .hwpx 파일로 저장."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # mimetype: 압축 없이 첫 번째로 저장 (HWPX 표준)
            zf.writestr(
                zipfile.ZipInfo("mimetype"),
                "application/hwp+zip".encode(),
                compress_type=zipfile.ZIP_STORED,
            )
            zf.writestr("META-INF/container.xml", self._build_container_xml())
            zf.writestr("Contents/content.hml", self._build_content_hml())

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
    doc.save(args.output)


if __name__ == "__main__":
    main()
