"""Shared GET/HEAD/Range-206 audio streaming, extracted from the demo-dataset
audio route so the new session-asset audio route serves bytes identically --
same headers, same partial-content behavior, same safe
`Content-Disposition` -- rather than a second, potentially divergent
implementation.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, StreamingResponse, Response


def stream_audio_file(path: Path, request: Request, safe_filename: str, media_type: str) -> Response:
    """Streams `path`'s bytes under the opaque `safe_filename` only -- never
    the real filesystem path or original filename. Honors a `Range` header
    with a real 206 partial-content response; falls back to a plain
    `FileResponse` (which itself serves both GET and HEAD) otherwise, or if
    the `Range` header is malformed."""

    file_size = path.stat().st_size

    range_header = request.headers.get("range")
    if range_header:
        try:
            range_value = range_header.replace("bytes=", "")
            start_str, _, end_str = range_value.partition("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1

            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            content_length = end - start + 1

            def generate_chunks():
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = handle.read(min(8192, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            return StreamingResponse(
                generate_chunks(),
                status_code=206,
                media_type=media_type,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Content-Length": str(content_length),
                    "Content-Disposition": f'inline; filename="{safe_filename}"',
                },
            )
        except (ValueError, IndexError):
            pass

    return FileResponse(
        path=path,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "Content-Disposition": f'inline; filename="{safe_filename}"',
        },
    )
