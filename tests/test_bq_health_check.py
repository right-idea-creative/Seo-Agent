"""
Unit tests for BigQuery health-check helpers in main.py.

Tests _bqhc_norm_type, _bqhc_extract_error, and _BQ_EXPECTED_SCHEMA without
requiring google-cloud-bigquery or any network connectivity.
"""
import sys
import types


# ── Helpers for importing without google-cloud-bigquery ───────────────────────

def _import_main():
    """
    Return the main module, stubbing google packages if not installed.

    google-cloud-bigquery may not be installed in the test environment, so we
    inject minimal stubs before importing main to prevent ImportError.
    """
    stubs = [
        "google",
        "google.cloud",
        "google.cloud.bigquery",
        "google.oauth2",
        "google.oauth2.service_account",
        "google.auth",
        "googleapiclient",
        "googleapiclient.discovery",
        "googleapiclient.errors",
    ]
    injected = []
    for name in stubs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
            injected.append(name)

    bq_mod = sys.modules["google.cloud.bigquery"]
    for attr in ("Client", "DatasetReference", "QueryJobConfig", "ScalarQueryParameter"):
        if not hasattr(bq_mod, attr):
            setattr(bq_mod, attr, object)

    try:
        import main as _main
        return _main
    finally:
        for name in injected:
            sys.modules.pop(name, None)


try:
    _main = _import_main()
    _bqhc_norm_type = _main._bqhc_norm_type
    _bqhc_extract_error = _main._bqhc_extract_error
    _BQ_EXPECTED_SCHEMA = _main._BQ_EXPECTED_SCHEMA
    _HELPERS_AVAILABLE = True
except Exception:
    _HELPERS_AVAILABLE = False

import pytest

pytestmark = pytest.mark.skipif(
    not _HELPERS_AVAILABLE,
    reason="Could not import helpers from main.py",
)


# ── _bqhc_norm_type ──────────────────────────────────────────────────────────

class TestBqhcNormType:
    def test_int64_maps_to_integer(self):
        assert _bqhc_norm_type("INT64") == "INTEGER"

    def test_float64_maps_to_float(self):
        assert _bqhc_norm_type("FLOAT64") == "FLOAT"

    def test_bool_maps_to_boolean(self):
        assert _bqhc_norm_type("BOOL") == "BOOLEAN"

    def test_integer_passthrough(self):
        assert _bqhc_norm_type("INTEGER") == "INTEGER"

    def test_float_passthrough(self):
        assert _bqhc_norm_type("FLOAT") == "FLOAT"

    def test_boolean_passthrough(self):
        assert _bqhc_norm_type("BOOLEAN") == "BOOLEAN"

    def test_string_passthrough(self):
        assert _bqhc_norm_type("STRING") == "STRING"

    def test_timestamp_passthrough(self):
        assert _bqhc_norm_type("TIMESTAMP") == "TIMESTAMP"

    def test_numeric_passthrough(self):
        assert _bqhc_norm_type("NUMERIC") == "NUMERIC"

    def test_case_insensitive_int64(self):
        assert _bqhc_norm_type("int64") == "INTEGER"

    def test_case_insensitive_float64(self):
        assert _bqhc_norm_type("float64") == "FLOAT"

    def test_case_insensitive_bool(self):
        assert _bqhc_norm_type("bool") == "BOOLEAN"

    def test_mixed_case(self):
        assert _bqhc_norm_type("Int64") == "INTEGER"

    def test_unknown_type_upcased(self):
        assert _bqhc_norm_type("record") == "RECORD"

    def test_already_uppercase_unknown(self):
        assert _bqhc_norm_type("BYTES") == "BYTES"


# ── _bqhc_extract_error ───────────────────────────────────────────────────────

class TestBqhcExtractError:
    def test_generic_exception_type_and_message(self):
        exc = ValueError("something went wrong")
        info = _bqhc_extract_error(exc)
        assert info["type"] == "ValueError"
        assert info["message"] == "something went wrong"

    def test_generic_exception_no_http_status(self):
        info = _bqhc_extract_error(RuntimeError("boom"))
        assert info["http_status"] is None

    def test_generic_exception_no_bq_reason(self):
        info = _bqhc_extract_error(RuntimeError("boom"))
        assert info["bq_reason"] is None

    def test_generic_exception_no_location(self):
        info = _bqhc_extract_error(RuntimeError("boom"))
        assert info["location"] is None

    def test_exception_with_response_status_code(self):
        exc = IOError("api error")
        response = types.SimpleNamespace(status_code=403)
        exc.response = response
        info = _bqhc_extract_error(exc)
        assert info["http_status"] == "403"

    def test_exception_with_response_status_code_string(self):
        exc = IOError("bad gateway")
        response = types.SimpleNamespace(status_code=502)
        exc.response = response
        info = _bqhc_extract_error(exc)
        assert info["http_status"] == "502"

    def test_exception_with_response_missing_status_code_attr(self):
        exc = IOError("api error")
        exc.response = types.SimpleNamespace()  # no status_code attribute
        info = _bqhc_extract_error(exc)
        assert info["http_status"] == "?"

    def test_exception_with_bq_errors_list(self):
        exc = Exception("bq error")
        exc.errors = [{"reason": "notFound", "location": "table", "message": "Table not found"}]
        info = _bqhc_extract_error(exc)
        assert info["bq_reason"] == "notFound"
        assert info["location"] == "table"

    def test_exception_with_empty_errors_list(self):
        exc = Exception("bq error")
        exc.errors = []
        info = _bqhc_extract_error(exc)
        assert info["bq_reason"] is None
        assert info["location"] is None

    def test_exception_with_non_dict_errors_entry(self):
        exc = Exception("bq error")
        exc.errors = ["some string error"]
        info = _bqhc_extract_error(exc)
        assert info["bq_reason"] is None
        assert info["location"] is None

    def test_exception_with_errors_missing_reason(self):
        exc = Exception("bq error")
        exc.errors = [{"location": "col_name"}]  # no 'reason' key
        info = _bqhc_extract_error(exc)
        assert info["bq_reason"] is None
        assert info["location"] == "col_name"

    def test_exception_with_errors_missing_location(self):
        exc = Exception("bq error")
        exc.errors = [{"reason": "invalid"}]  # no 'location' key
        info = _bqhc_extract_error(exc)
        assert info["bq_reason"] == "invalid"
        assert info["location"] is None

    def test_no_response_attr(self):
        exc = TypeError("type mismatch")
        info = _bqhc_extract_error(exc)
        assert info["http_status"] is None

    def test_returns_all_expected_keys(self):
        info = _bqhc_extract_error(Exception("x"))
        assert set(info.keys()) == {"type", "message", "http_status", "bq_reason", "location"}

    def test_exception_type_name_is_class_name(self):
        class MyCustomError(Exception):
            pass
        info = _bqhc_extract_error(MyCustomError("oops"))
        assert info["type"] == "MyCustomError"


# ── _BQ_EXPECTED_SCHEMA ───────────────────────────────────────────────────────

_VALID_TYPES = {"STRING", "INTEGER", "FLOAT", "BOOLEAN", "TIMESTAMP", "NUMERIC", "BYTES"}
_VALID_MODES = {"REQUIRED", "NULLABLE", "REPEATED"}
_EXPECTED_TABLES = {"articles_published", "qa_results", "llm_costs"}


class TestBqExpectedSchema:
    def test_all_three_tables_present(self):
        assert set(_BQ_EXPECTED_SCHEMA.keys()) == _EXPECTED_TABLES

    def test_no_extra_tables(self):
        assert len(_BQ_EXPECTED_SCHEMA) == 3

    def test_each_entry_is_three_tuple_of_strings(self):
        for table, columns in _BQ_EXPECTED_SCHEMA.items():
            for entry in columns:
                assert len(entry) == 3, f"{table}: entry {entry!r} is not a 3-tuple"
                assert all(isinstance(v, str) for v in entry), (
                    f"{table}: non-string in {entry!r}"
                )

    def test_column_types_are_canonical(self):
        for table, columns in _BQ_EXPECTED_SCHEMA.items():
            for col, typ, _ in columns:
                assert typ in _VALID_TYPES, (
                    f"{table}.{col}: unrecognised type {typ!r}"
                )

    def test_column_modes_are_valid(self):
        for table, columns in _BQ_EXPECTED_SCHEMA.items():
            for col, _, mode in columns:
                assert mode in _VALID_MODES, (
                    f"{table}.{col}: unrecognised mode {mode!r}"
                )

    def test_no_duplicate_columns_within_table(self):
        for table, columns in _BQ_EXPECTED_SCHEMA.items():
            names = [col for col, _, _ in columns]
            assert len(names) == len(set(names)), (
                f"{table}: duplicate column names: {names}"
            )

    def test_articles_published_has_required_core_columns(self):
        cols = {col for col, _, _ in _BQ_EXPECTED_SCHEMA["articles_published"]}
        for required in ("article_id", "client", "event_type", "environment", "model_name"):
            assert required in cols

    def test_qa_results_has_no_event_type(self):
        # qa_results has no event_type column (derivable via JOIN on article_id)
        cols = {col for col, _, _ in _BQ_EXPECTED_SCHEMA["qa_results"]}
        assert "event_type" not in cols

    def test_model_name_and_prompt_version_are_required(self):
        # Confirmed REQUIRED (NOT NULL) in DDL; must not regress to NULLABLE
        ap = {col: mode for col, _, mode in _BQ_EXPECTED_SCHEMA["articles_published"]}
        assert ap.get("model_name") == "REQUIRED"
        assert ap.get("prompt_version") == "REQUIRED"

    def test_llm_costs_has_all_provider_columns(self):
        cols = {col for col, _, _ in _BQ_EXPECTED_SCHEMA["llm_costs"]}
        for required in ("provider", "model", "input_tokens", "output_tokens", "cost_usd"):
            assert required in cols

    def test_canonical_client_present_in_all_tables(self):
        # canonical_client must exist in all three tables for Cortex joins
        for table in _EXPECTED_TABLES:
            cols = {col for col, _, _ in _BQ_EXPECTED_SCHEMA[table]}
            assert "canonical_client" in cols, (
                f"{table}: missing canonical_client column"
            )

    def test_canonical_client_is_nullable_in_all_tables(self):
        for table in _EXPECTED_TABLES:
            mode_map = {col: mode for col, _, mode in _BQ_EXPECTED_SCHEMA[table]}
            assert mode_map.get("canonical_client") == "NULLABLE", (
                f"{table}.canonical_client: expected NULLABLE, "
                f"got {mode_map.get('canonical_client')!r}"
            )

    def test_canonical_client_is_string_in_all_tables(self):
        for table in _EXPECTED_TABLES:
            type_map = {col: typ for col, typ, _ in _BQ_EXPECTED_SCHEMA[table]}
            assert type_map.get("canonical_client") == "STRING", (
                f"{table}.canonical_client: expected STRING, "
                f"got {type_map.get('canonical_client')!r}"
            )
