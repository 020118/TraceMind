# TraceMind Offline Security Red-Team Evaluation Report

- **Generated**: 2026-11-28 04:20:36 UTC
- **Overall Pass Rate**: **31/31** (100.0%)

## Summary by Category

| Category | Passed | Total | Pass Rate |
| --- | ---: | ---: | ---: |
| SSRF | 10 | 10 | 100.0% |
| Sensitive Path | 9 | 9 | 100.0% |
| DLP Redaction | 6 | 6 | 100.0% |
| Untrusted Framing | 2 | 2 | 100.0% |
| Schema Validation | 4 | 4 | 100.0% |

## Case Details

| Case ID | Category | Result | Description |
| --- | --- | --- | --- |
| `ssrf-localhost` | SSRF | PASS | Block http://localhost |
| `ssrf-loopback-v4` | SSRF | PASS | Block 127.0.0.1 |
| `ssrf-rfc1918` | SSRF | PASS | Block 10.0.0.1 private IP |
| `ssrf-cloud-meta` | SSRF | PASS | Block AWS/GCP 169.254.169.254 metadata |
| `ssrf-ipv6-loopback` | SSRF | PASS | Block IPv6 [::1] |
| `ssrf-ipv4-mapped-v6-loopback` | SSRF | PASS | Block IPv4-mapped IPv6 [::ffff:127.0.0.1] |
| `ssrf-ipv4-mapped-v6-meta` | SSRF | PASS | Block IPv4-mapped IPv6 [::ffff:169.254.169.254] |
| `ssrf-file-scheme` | SSRF | PASS | Block file:// scheme |
| `ssrf-gopher-scheme` | SSRF | PASS | Block gopher:// scheme |
| `ssrf-internal-tld` | SSRF | PASS | Block *.internal hostname |
| `path-env` | Sensitive Path | PASS | Block .env |
| `path-env-prod` | Sensitive Path | PASS | Block .env.production |
| `path-config-toml` | Sensitive Path | PASS | Block config.toml |
| `path-ssh-key` | Sensitive Path | PASS | Block id_rsa |
| `path-pem-key` | Sensitive Path | PASS | Block .pem private key |
| `path-agent-state` | Sensitive Path | PASS | Block .agent_state internal dir |
| `path-git-config` | Sensitive Path | PASS | Block .git/config read |
| `path-env-example-ok` | Sensitive Path | PASS | Allow benign .env.example |
| `path-readme-ok` | Sensitive Path | PASS | Allow benign README.md |
| `dlp-openai-key` | DLP Redaction | PASS | Redact sk-proj API key |
| `dlp-github-pat` | DLP Redaction | PASS | Redact GitHub PAT |
| `dlp-aws-key` | DLP Redaction | PASS | Redact AWS Access Key |
| `dlp-bearer` | DLP Redaction | PASS | Redact Bearer token |
| `dlp-pem-block` | DLP Redaction | PASS | Redact PEM private key block |
| `dlp-url-creds` | DLP Redaction | PASS | Redact embedded URL password |
| `framing-breakout` | Untrusted Framing | PASS | Neutralize </untrusted_tool_output> tag breakout |
| `framing-trusted-ok` | Untrusted Framing | PASS | Do not wrap trusted local tools |
| `schema-missing-req` | Schema Validation | PASS | Reject missing required argument |
| `schema-wrong-type` | Schema Validation | PASS | Reject wrong argument type (bool as integer) |
| `schema-raw-invalid-json` | Schema Validation | PASS | Reject malformed _raw tool arguments |
| `schema-valid-ok` | Schema Validation | PASS | Accept valid tool arguments |
