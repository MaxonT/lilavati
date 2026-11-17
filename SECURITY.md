# Security Policy

Līlāvatī is alpha software. Please report vulnerabilities **privately** via GitHub's
"Report a vulnerability" (Security → Advisories) on this repository rather than a public issue.
We aim to acknowledge reports within 7 days.

Scope notes:
* The bundled FastAPI server has no built-in authentication and binds to localhost by default. Exposing it publicly without an authenticating proxy is a deployment issue, not a vulnerability.
* Never commit API keys; LLM provider keys are read from environment variables.
* Model/engine outputs are not financial advice and have not been independently audited.
