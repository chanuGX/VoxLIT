import re

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from .settings import settings
from .redis import ensure_session

_SID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class InvalidSessionId(Exception):
    """Raised when a session id fails the exact-format check."""


def validate_session_id(sid: str | None) -> str:
    """Full-match only against the exact 32-lowercase-hex-char shape
    `ensure_session()` mints (`uuid.uuid4().hex`) -- a value that doesn't
    match exactly is rejected outright, never re-cased or "fixed up"."""

    if not sid or not _SID_PATTERN.fullmatch(sid):
        raise InvalidSessionId("Session id is missing or malformed.")
    return sid


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        sid = await ensure_session(request.cookies.get(settings.SESSION_COOKIE_NAME))
        request.state.sid = sid
        resp: Response = await call_next(request)
        if settings.SESSION_COOKIE_NAME not in request.cookies:
            resp.set_cookie(
                settings.SESSION_COOKIE_NAME, sid,
                max_age=settings.SESSION_TTL_SECONDS,
                httponly=True, secure=settings.COOKIE_SECURE,
                samesite=settings.COOKIE_SAMESITE, domain=settings.COOKIE_DOMAIN, path="/",
            )
        return resp
