import pytest

from governance.pii import detect_pii, luhn_valid, redact, verhoeff_valid


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


@pytest.mark.parametrize(
    "number,valid", [("2363", True), ("2364", False), ("452188301942", True), ("452188301945", False)]
)
def test_verhoeff(number, valid):
    assert verhoeff_valid(number) is valid


@pytest.mark.parametrize(
    "text,types",
    [
        ("mail a.b@example.com", ["email"]),
        ("call +91 98450 12345", ["phone"]),
        ("Aadhaar 4521 8830 1942", ["aadhaar"]),
        ("Aadhaar-like 4521 8830 1945 with bad checksum", []),
        ("PAN ABCPE1234K", ["pan"]),
        ("card 4111-1111-1111-1111", ["credit_card"]),
        ("card 4111 1111 1111 1112 fails Luhn", []),
        ("room 404, 15000 rupees, v2.14.3", []),
    ],
)
def test_detection(text, types):
    assert [f.type for f in detect_pii(text)] == types


def test_card_is_not_double_counted_as_phone_or_aadhaar():
    assert [f.type for f in detect_pii("4111 1111 1111 1111")] == ["credit_card"]


def test_severity():
    sev = {f.type: f.severity for f in detect_pii("x@y.com 4111 1111 1111 1111")}
    assert sev == {"email": "medium", "credit_card": "high"}


def test_redact_replaces_every_finding():
    out = redact("Email a@b.com or call 9845012345.")
    assert out == "Email [REDACTED_EMAIL] or call [REDACTED_PHONE]."
