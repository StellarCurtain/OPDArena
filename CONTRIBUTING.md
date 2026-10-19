# Contributing

Use a separate environment and install the pinned runtime with `scripts/install.sh`.
Keep project logic in `opdarena/`, add usage examples when introducing a public
option, and keep generated data, model weights, results, and credentials out of Git.

Run `pytest`, `ruff check`, `ruff format --check`, and `scripts/check_release.py`
before proposing a change. Changes to distributed training should also run the
small GPU workflow in `docs/validation.md`; document the GPU count and tested
methods.

Use English in code comments and public documentation. Preserve upstream
copyright and license notices. Explain changes to vendored code in
`THIRD_PARTY.md` and add a short modification notice to the affected source file.

Use small, focused commits. Include the problem, the resulting behavior, and
validation in pull requests. Never include access tokens or private data in an
issue, commit, or training log attached to a report. Contributions are made under
the repository's Apache-2.0 license.
