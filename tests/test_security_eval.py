from __future__ import annotations

from eval.security_eval import render_markdown_report, run_security_eval


def test_security_redteam_eval_all_cases_pass():
    results = run_security_eval()
    assert len(results) >= 25
    failures = [r for r in results if not r.passed]
    assert not failures, f"Security eval failures: {failures}"

    report = render_markdown_report(results)
    assert "Overall Pass Rate" in report
    assert "SSRF" in report
    assert "DLP Redaction" in report
    assert "Untrusted Framing" in report
