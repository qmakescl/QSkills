#!/usr/bin/env python3
"""
pack.py — 언팩된 디렉토리를 .hwpx 파일로 다시 패킹한다.

사용법:
    python scripts/pack.py unpacked/ output.hwpx
    python scripts/pack.py unpacked/ output.hwpx --original document.hwpx
"""

import sys
import zipfile
import argparse
from pathlib import Path


# mimetype은 압축 없이 맨 앞에 저장해야 HWPX 표준에 맞다
MIMETYPE_FILE = "mimetype"
MIMETYPE_VALUE = "application/hwp+zip"


def pack(input_dir: Path, output_path: Path, original_hwpx: Path = None):
    """디렉토리를 .hwpx로 팩."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. mimetype을 압축 없이 첫 번째로 저장 (ZIP 표준)
        mimetype_path = input_dir / MIMETYPE_FILE
        if mimetype_path.exists():
            zf.write(mimetype_path, MIMETYPE_FILE,
                     compress_type=zipfile.ZIP_STORED)
        else:
            zf.writestr(zipfile.ZipInfo(MIMETYPE_FILE),
                        MIMETYPE_VALUE.encode(),
                        compress_type=zipfile.ZIP_STORED)

        # 2. 나머지 파일 추가
        for file_path in sorted(input_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(input_dir)
            # mimetype은 이미 추가했으므로 스킵
            if str(rel) == MIMETYPE_FILE:
                continue
            zf.write(file_path, str(rel))

    print(f"✅ 팩 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="디렉토리 → .hwpx 패킹")
    parser.add_argument("input_dir", help="언팩된 디렉토리 경로")
    parser.add_argument("output_file", help="출력 .hwpx 파일 경로")
    parser.add_argument("--original", help="원본 .hwpx (메타데이터 참조용, 선택)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"오류: 디렉토리 없음: {input_dir}", file=sys.stderr)
        sys.exit(1)

    original = Path(args.original) if args.original else None
    pack(input_dir, Path(args.output_file), original)


if __name__ == "__main__":
    main()
