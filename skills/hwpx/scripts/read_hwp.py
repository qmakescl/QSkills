#!/usr/bin/env python3
"""
read_hwp.py — .hwp / .hwpx 파일에서 텍스트를 추출한다.

사용법:
    python scripts/read_hwp.py document.hwp
    python scripts/read_hwp.py document.hwpx
    python scripts/read_hwp.py document.hwp --output out.txt

처리 방식:
    .hwpx → python-hwpx TextExtractor (우선)
    .hwp  → pyhwp → LibreOffice (폴백 순서)
"""

import sys
import argparse
import subprocess
from pathlib import Path


def read_with_hwpx(hwp_path: Path) -> str:
    """python-hwpx TextExtractor로 .hwpx 텍스트 추출."""
    from hwpx import TextExtractor
    te = TextExtractor(str(hwp_path))
    text = te.extract_text()
    if not text:
        raise RuntimeError("TextExtractor 출력이 비어있음")
    return text


def read_with_pyhwp(hwp_path: Path) -> str:
    """pyhwp 직접 API 방식으로 텍스트 추출.
    stdout.buffer 캡처 문제를 피하기 위해 임시 파일에 직접 기록한다."""
    import tempfile
    from contextlib import closing
    from hwp5.hwp5txt import TextTransform
    from hwp5.xmlmodel import Hwp5File

    text_transform = TextTransform()
    transform = text_transform.transform_hwp5_to_text

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with closing(Hwp5File(str(hwp_path))) as hwp5file:
            with open(tmp_path, "wb") as dest:
                transform(hwp5file, dest)
        text = tmp_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            raise RuntimeError("pyhwp 출력이 비어있음")
        return text
    finally:
        tmp_path.unlink(missing_ok=True)


def read_with_libreoffice(hwp_path: Path) -> str:
    """LibreOffice로 txt 변환 후 텍스트 반환 (pyhwp 대체 수단)."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "txt:Text",
             "--outdir", tmp, str(hwp_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice 변환 실패: {result.stderr}")
        txt_file = Path(tmp) / (hwp_path.stem + ".txt")
        if not txt_file.exists():
            raise RuntimeError("변환된 txt 파일을 찾을 수 없음")
        return txt_file.read_text(encoding="utf-8", errors="replace")


def extract_text(hwp_path: Path) -> str:
    """hwp/hwpx 파일에서 텍스트 추출.

    .hwpx: python-hwpx TextExtractor 사용
    .hwp:  pyhwp → LibreOffice 순으로 폴백
    """
    # .hwpx는 python-hwpx TextExtractor 전용
    if hwp_path.suffix.lower() == ".hwpx":
        try:
            return read_with_hwpx(hwp_path)
        except Exception as e:
            raise RuntimeError(
                f".hwpx 텍스트 추출 실패: {e}\n"
                f"설치 방법: pip install python-hwpx --break-system-packages"
            )

    # .hwp: pyhwp 우선, LibreOffice 폴백
    pyhwp_err = None
    try:
        return read_with_pyhwp(hwp_path)
    except Exception as e:
        pyhwp_err = e
        print(f"[INFO] pyhwp 실패, LibreOffice로 재시도: {e}", file=sys.stderr)

    try:
        return read_with_libreoffice(hwp_path)
    except Exception as lo_err:
        raise RuntimeError(
            f"텍스트 추출 실패.\n"
            f"  pyhwp 오류: {pyhwp_err}\n"
            f"  LibreOffice 오류: {lo_err}\n\n"
            f"pyhwp 설치 방법: pip install pyhwp --break-system-packages"
        )


def main():
    parser = argparse.ArgumentParser(description=".hwp / .hwpx 파일 텍스트 추출")
    parser.add_argument("hwp_file", help=".hwp 또는 .hwpx 파일 경로")
    parser.add_argument("--output", "-o", help="출력 파일 경로 (없으면 stdout)")
    args = parser.parse_args()

    hwp_path = Path(args.hwp_file)
    if not hwp_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {hwp_path}", file=sys.stderr)
        sys.exit(1)

    text = extract_text(hwp_path)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"저장 완료: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
