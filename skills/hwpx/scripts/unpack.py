#!/usr/bin/env python3
"""
unpack.py — .hwpx 파일을 디렉토리로 언팩하고 XML을 들여쓰기 처리한다.

사용법:
    python scripts/unpack.py document.hwpx unpacked/
"""

import sys
import zipfile
import argparse
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


def pretty_print_xml(xml_bytes: bytes) -> str:
    """XML을 들여쓰기 포맷으로 변환. 파싱 실패 시 원본 반환."""
    try:
        # lxml 우선 (더 나은 네임스페이스 처리)
        from lxml import etree
        root = etree.fromstring(xml_bytes)
        return etree.tostring(root, pretty_print=True, encoding="unicode",
                              xml_declaration=False)
    except Exception:
        pass

    # stdlib fallback
    try:
        ET.indent  # Python 3.9+
        root = ET.fromstring(xml_bytes)
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")
    except Exception:
        pass

    # 그대로 반환
    return xml_bytes.decode("utf-8", errors="replace")


def unpack(hwpx_path: Path, output_dir: Path):
    """hwpx를 output_dir에 언팩."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    XML_EXTENSIONS = {".xml", ".hpf", ".rels"}

    with zipfile.ZipFile(hwpx_path, "r") as zf:
        for member in zf.namelist():
            dest = output_dir / member
            if member.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(member)

            # XML 파일은 들여쓰기 처리
            if Path(member).suffix.lower() in XML_EXTENSIONS:
                try:
                    formatted = pretty_print_xml(data)
                    dest.write_text(formatted, encoding="utf-8")
                    continue
                except Exception:
                    pass  # 실패 시 그냥 저장

            dest.write_bytes(data)

    print(f"✅ 언팩 완료: {hwpx_path} → {output_dir}")
    print(f"   본문 XML: {output_dir}/Contents/section0.xml")
    print(f"   스타일:   {output_dir}/Contents/header.xml")


def main():
    parser = argparse.ArgumentParser(description=".hwpx 파일 언팩")
    parser.add_argument("hwpx_file", help=".hwpx 파일 경로")
    parser.add_argument("output_dir", help="출력 디렉토리 경로")
    args = parser.parse_args()

    hwpx_path = Path(args.hwpx_file)
    if not hwpx_path.exists():
        print(f"오류: 파일 없음: {hwpx_path}", file=sys.stderr)
        sys.exit(1)

    unpack(hwpx_path, Path(args.output_dir))


if __name__ == "__main__":
    main()
