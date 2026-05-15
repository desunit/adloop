"""Tests for server error formatting and MCP instructions field."""

import pytest

from adloop.ads.gaql import _parse_gaql_error
from adloop.server import (
    _build_orchestration_instructions,
    _coerce_json_string_to_list,
    _structured_error,
)


def test_structured_error_detects_invalid_developer_token():
    error = Exception(
        "errors { error_code { authentication_error: DEVELOPER_TOKEN_INVALID } "
        'message: "The developer token is not valid." }'
    )

    result = _structured_error("list_accounts", error)

    assert result["error"] == "Google Ads authentication failed — developer token is invalid."
    assert result["auth_error"] == "DEVELOPER_TOKEN_INVALID"
    assert "ads.developer_token" in result["hint"]


def test_structured_error_detects_test_only_developer_token():
    error = Exception(
        "errors { error_code { authorization_error: DEVELOPER_TOKEN_NOT_APPROVED } "
        'message: "The developer token is only approved for use with test accounts." }'
    )

    result = _structured_error("list_accounts", error)

    assert result["error"] == (
        "Google Ads authorization failed — developer token is not approved "
        "for production accounts."
    )
    assert result["auth_error"] == "DEVELOPER_TOKEN_NOT_APPROVED"
    assert "test accounts" in result["hint"]


def test_structured_error_detects_revoked_oauth_token():
    error = Exception("invalid_grant: Token has been expired or revoked.")

    result = _structured_error("health_check", error)

    assert result["error"] == "Authentication failed — OAuth token expired or revoked."
    assert result["auth_error"] == "INVALID_GRANT"
    assert "~/.adloop/token.json" in result["hint"]


def test_parse_gaql_error_detects_invalid_developer_token():
    error = Exception(
        "errors { error_code { authentication_error: DEVELOPER_TOKEN_INVALID } "
        'message: "The developer token is not valid." }'
    )

    result = _parse_gaql_error(error)

    assert result.startswith("DEVELOPER_TOKEN_INVALID:")
    assert "ads.developer_token" in result


def test_parse_gaql_error_detects_test_only_developer_token():
    error = Exception(
        "errors { error_code { authorization_error: DEVELOPER_TOKEN_NOT_APPROVED } "
        'message: "The developer token is only approved for use with test accounts." }'
    )

    result = _parse_gaql_error(error)

    assert result.startswith("DEVELOPER_TOKEN_NOT_APPROVED:")
    assert "test accounts" in result


# ---------------------------------------------------------------------------
# MCP InitializeResult.instructions — orchestration hint
# ---------------------------------------------------------------------------


class TestOrchestrationInstructions:
    """The MCP `instructions` field is sent during the initialize handshake
    and (per spec) MAY be injected into the LLM's system prompt by clients
    that honor it. We send a compact must-knows summary, NOT the full ruleset.
    """

    def test_instructions_cover_safety_essentials(self):
        text = _build_orchestration_instructions()

        # Must mention the two-step write pattern.
        assert "PREVIEW" in text or "preview" in text
        assert "plan_id" in text
        assert "confirm_and_apply" in text

        # Must mention dry_run defaults.
        assert "dry_run" in text

    def test_instructions_cover_pre_write_checks(self):
        text = _build_orchestration_instructions()
        # Most expensive mistake we want to prevent.
        assert "BROAD" in text and "Smart Bidding" in text
        # Second-most expensive: dead URLs.
        assert "final_url" in text or "URL" in text
        # Avoid throwing budget at broken tracking.
        assert "tracking" in text.lower() or "conversions" in text.lower()

    def test_instructions_cover_data_literacy(self):
        text = _build_orchestration_instructions()
        # GDPR consent gap is the #1 source of misdiagnosed "tracking issues".
        assert "GDPR" in text or "consent" in text.lower()
        # Geo / language targeting is mandatory.
        assert "geo" in text.lower()
        assert "language" in text.lower()

    def test_instructions_point_at_full_ruleset(self):
        text = _build_orchestration_instructions()
        # The hint should tell the model where the full rules live.
        assert "install-rules" in text or "adloop.mdc" in text or "CLAUDE.md" in text

    def test_instructions_are_compact_not_full_rules(self):
        # Spec describes `instructions` as a "hint" — not a manual.
        # Hard cap so we don't accidentally regress to dumping 50KB through
        # the handshake.
        text = _build_orchestration_instructions()
        assert len(text) < 5_000, (
            f"instructions field is {len(text)} bytes — should be a compact "
            f"hint (<5KB). For full rules use install-rules."
        )

    def test_instructions_are_attached_to_mcp_server(self):
        # FastMCP forwards the constructor arg into the wire-protocol
        # InitializeResult.instructions. Sanity check it's actually wired.
        from adloop.server import mcp

        instructions = getattr(mcp, "instructions", None)
        assert instructions is not None
        assert isinstance(instructions, str)
        assert len(instructions) > 100, (
            "MCP instructions field should be the compact orchestration hint, "
            "not a placeholder"
        )


class TestListParamJsonStringCoercion:
    """Regression for issue #28 — some MCP clients (e.g. Cowork) serialize
    list-typed tool arguments as JSON-encoded strings rather than native
    arrays, causing Pydantic to reject the call with ``Input should be a
    valid list``. ``_coerce_json_string_to_list`` runs as a ``BeforeValidator``
    on every tool param annotated with ``_StrList``/``_StrListOpt``/
    ``_DictList``/``_DictListOpt`` and decodes the JSON before the standard
    list validator runs.
    """

    def test_passes_through_native_list(self):
        assert _coerce_json_string_to_list(["a", "b"]) == ["a", "b"]

    def test_passes_through_none(self):
        assert _coerce_json_string_to_list(None) is None

    def test_decodes_json_encoded_str_list_of_strings(self):
        assert _coerce_json_string_to_list('["pagePath"]') == ["pagePath"]

    def test_decodes_json_encoded_str_list_with_multiple_metrics(self):
        encoded = '["sessions", "totalUsers", "screenPageViews", "bounceRate"]'
        assert _coerce_json_string_to_list(encoded) == [
            "sessions",
            "totalUsers",
            "screenPageViews",
            "bounceRate",
        ]

    def test_decodes_json_encoded_str_list_of_dicts(self):
        encoded = '[{"text": "shoes", "match_type": "EXACT"}]'
        assert _coerce_json_string_to_list(encoded) == [
            {"text": "shoes", "match_type": "EXACT"}
        ]

    def test_non_list_json_string_passes_through_unchanged(self):
        # ``'42'`` and ``'"pagePath"'`` decode to non-list values. We must
        # NOT pretend they're lists — let Pydantic's normal list validator
        # raise the standard error so the client sees a useful schema
        # violation rather than a silently-mangled payload.
        assert _coerce_json_string_to_list("42") == "42"
        assert _coerce_json_string_to_list('"pagePath"') == '"pagePath"'

    def test_invalid_json_string_passes_through_unchanged(self):
        # Bare strings like ``pagePath`` aren't valid JSON. Pass them through
        # so Pydantic emits the standard ``Input should be a valid list``
        # error (which is the correct response — that input is genuinely
        # not a list, JSON-encoded or otherwise).
        assert _coerce_json_string_to_list("pagePath") == "pagePath"

    def test_empty_string_passes_through_unchanged(self):
        assert _coerce_json_string_to_list("") == ""


class TestListParamJsonStringCoercionEndToEnd:
    """End-to-end: the BeforeValidator must fire through Pydantic when the
    annotated types are used in a model. If this ever breaks (e.g. because
    we accidentally drop the ``Annotated`` wrapper), the unit-level tests
    above would still pass but the real-world fix would be silently gone.
    """

    def _model_with_str_list(self):
        from pydantic import BaseModel

        from adloop.server import _StrList, _StrListOpt

        class _M(BaseModel):
            required: _StrList
            optional: _StrListOpt = None

        return _M

    def test_pydantic_accepts_json_encoded_string_for_required_list(self):
        model_cls = self._model_with_str_list()
        m = model_cls(required='["pagePath"]')
        assert m.required == ["pagePath"]

    def test_pydantic_accepts_json_encoded_string_for_optional_list(self):
        model_cls = self._model_with_str_list()
        m = model_cls(required=["a"], optional='["b", "c"]')
        assert m.optional == ["b", "c"]

    def test_pydantic_still_rejects_non_list_strings(self):
        model_cls = self._model_with_str_list()
        with pytest.raises(Exception, match="valid list"):
            model_cls(required="pagePath")  # bare string, not JSON

    def test_pydantic_still_accepts_native_arrays(self):
        model_cls = self._model_with_str_list()
        m = model_cls(required=["a", "b"])
        assert m.required == ["a", "b"]
