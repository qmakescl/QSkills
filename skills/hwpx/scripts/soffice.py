#!/usr/bin/env python3
"""
soffice.py — LibreOffice를 이용한 파일 변환 래퍼.

사용법:
    python scripts/soffice.py --headless --convert-to pdf document.hwp
    python scripts/soffice.py --headless --convert-to docx document.hwpx
    python scripts/soffice.py --headless --convert-to pdf --outdir ./output/ document.hwp

지원 변환:
    .hwp  → pdf, docx, txt, odt
    .hwpx → pdf, docx, txt, odt
    .docx → hwpx (LibreOffice HWP 필터 설치 시)
"""

import sys
import subprocess
import argparse
from pathlib import Path


LIBREOFFICE_PATHS = [
    "/usr/bin/libreoffice",
    "/usr/bin/soffice",
    "/opt/libreoffice/program/soffice",
]


def find_libreoffice() -> str:
    """사용 가능한 LibreOffice 실행 경로 반환."""
    for path in LIBREOFFICE_PATHS:
        if Path(path).exists():
            return path
    raise FileNotFoundError(
        "LibreOffice를 찾을 수 없습니다.\n"
        "설치 방법: sudo apt-get install libreoffice"
    )


def convert(
    input_file: Path,
    convert_to: str,
    outdir: Path = None,
    headless: bool = True,
) -> Path:
    """
    파일 변환 실행.

    Args:
        input_file:  변환할 파일 경로
        convert_to:  목표 포맷 (pdf, docx, txt, odt 등)
        outdir:      출력 디렉토리 (없으면 입력 파일과 같은 위치)
        headless:    GUI 없이 실행 여부

    Returns:
        변환된 파일 경로
    """
    soffice = find_libreoffice()
    out_path = outdir or input_file.parent
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [soffice]
    if headless:
        cmd.append("--headless")
    cmd += ["--convert-to", convert_to, "--outdir", str(out_path), str(input_file)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice 변환 실패 (exit {result.returncode})\n"
            f"stderr: {result.stderr.strip()}"
        )

    # 변환된 파일 경로 계산
    converted = out_path / (input_file.stem + "." + convert_to.split(":")[0])
    if converted.exists():
        print(f"✅ 변환 완료: {converted}")
        return converted
    else:
        # LibreOffice가 다른 확장자를 사용했을 수도 있음
        print(f"✅ 변환 완료 (출력 디렉토리: {out_path})")
        print(f"   stdout: {result.stdout.strip()}")
        return out_path


def main():
    parser = argparse.ArgumentParser(
        description="LibreOffice 파일 변환 래퍼",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="변환할 파일 경로")
    parser.add_argument("--convert-to", required=True,
                        help="목표 포맷 (pdf, docx, txt, odt 등)")
    parser.add_argument("--outdir", help="출력 디렉토리 (기본: 입력 파일과 같은 위치)")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="GUI 없이 실행 (기본: True)")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"오류: 파일 없음: {input_file}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else None
    convert(input_file, args.convert_to, outdir, args.headless)


if __name__ == "__main__":
    main()
