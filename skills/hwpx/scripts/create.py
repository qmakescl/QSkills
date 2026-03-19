#!/usr/bin/env python3
"""
create.py — python-hwpx 라이브러리를 사용해 HWPX 문서를 생성한다.

사용법:
    python scripts/create.py --output output.hwpx

Python API:
    from scripts.create import HWPXDocument

    doc = HWPXDocument()
    doc.add_paragraph("제목", style="title")    # 개요 1 스타일
    doc.add_paragraph("소제목", style="h2")     # 개요 2 스타일
    doc.add_paragraph("본문 내용입니다.")        # 바탕글 스타일 (기본)
    doc.add_blank_line()                         # 빈 줄
    doc.save("output.hwpx")

스타일 이름 (style=):
    "normal" / "바탕글"  → style_id_ref=0 (기본)
    "body"   / "본문"    → style_id_ref=1
    "title"  / "h1"     → style_id_ref=2 (개요 1)
    "h2"                 → style_id_ref=3 (개요 2)
    "h3"                 → style_id_ref=4 (개요 3)

의존성:
    pip install python-hwpx --break-system-packages
"""

import sys
import argparse
from pathlib import Path

try:
    from hwpx import HwpxDocument
except ImportError:
    print("오류: python-hwpx 미설치. 다음 명령으로 설치하세요:", file=sys.stderr)
    print("  pip install python-hwpx --break-system-packages", file=sys.stderr)
    sys.exit(1)


# 스타일 이름 → style_id_ref 매핑
# (HwpxDocument.new()의 기본 템플릿 스타일 기준)
STYLE_MAP = {
    # 한국어 이름
    "바탕글": 0,
    "본문": 1,
    "개요1": 2,
    "개요2": 3,
    "개요3": 4,
    "개요4": 5,
    # 영어 별칭
    "normal": 0,
    "body": 1,
    "title": 2,
    "h1": 2,
    "h2": 3,
    "h3": 4,
    "h4": 5,
    "outline1": 2,
    "outline2": 3,
    "outline3": 4,
}


class HWPXDocument:
    """python-hwpx 기반 HWPX 문서 생성 래퍼.

    python-hwpx 라이브러리(pip install python-hwpx)의 HwpxDocument.new()를
    래핑하여 간편한 문서 생성 인터페이스를 제공한다.
    생성된 파일은 한컴오피스(한글)에서 정상적으로 열린다.
    """

    def __init__(self):
        self._doc = HwpxDocument.new()

    def add_paragraph(
        self,
        text: str = "",
        *,
        style: str = None,
        style_id: int = None,
    ):
        """문단을 추가한다.

        Args:
            text:     문단 텍스트
            style:    스타일 이름 문자열 (STYLE_MAP 참조)
            style_id: style_id_ref 정수값 (style보다 낮은 우선순위)

        Returns:
            self (메서드 체이닝 지원)
        """
        sid = style_id  # 기본값

        if style is not None:
            # 대소문자 구분 없이 검색
            key_lower = style.lower()
            sid = STYLE_MAP.get(key_lower, STYLE_MAP.get(style, 0))

        self._doc.add_paragraph(text, style_id_ref=sid)
        return self

    def add_blank_line(self):
        """빈 줄(빈 문단)을 추가한다."""
        self._doc.add_paragraph("")
        return self

    def save(self, path):
        """HWPX 파일로 저장한다.

        Args:
            path: 출력 파일 경로 (str 또는 Path)
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._doc.save_to_path(str(out))
        print(f"✅ 저장 완료: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="HWPX 문서 생성 (python-hwpx 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", "-o", default="output.hwpx", help="출력 파일 경로")
    args = parser.parse_args()

    # 데모 문서 생성
    doc = HWPXDocument()
    doc.add_paragraph("HWPX 문서 제목", style="title")
    doc.add_blank_line()
    doc.add_paragraph("1. 첫 번째 소제목", style="h2")
    doc.add_paragraph("여기에 본문 내용을 입력합니다. python-hwpx 라이브러리로 생성된 문서입니다.")
    doc.add_blank_line()
    doc.add_paragraph("2. 두 번째 소제목", style="h2")
    doc.add_paragraph("두 번째 소제목의 본문 내용입니다.")
    doc.save(args.output)


if __name__ == "__main__":
    main()
