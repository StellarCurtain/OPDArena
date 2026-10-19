"""Check tracked-source hygiene without printing any matched credential values."""

import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "outputs",
    "models",
}
TEXT = {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".sh", ".txt", ".json", ".jinja"}
CHECKS = {
    "personal filesystem path": re.compile(r"/(?:home|mnt|scratch|Users)/[\w.-]+/"),
    "private-key material": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access credential": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9]{32,})\b"
    ),
    "signed or credentialed URL": re.compile(
        r'https?://[^\s"<>]+[?&](?:sig|access_token)=[A-Za-z0-9%+/=_-]{16,}'
    ),
}


def check():
    issues = []
    count = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(
            part in SKIP or part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if path.suffix not in TEXT and path.name not in ("LICENSE", "NOTICE", "Notice.txt"):
            continue
        count += 1
        source = path.read_text(encoding="utf-8")
        # Exclude the scanner's own pattern definitions from credential matching.
        checks = {} if relative == Path("scripts/check_release.py") else CHECKS
        for kind, pattern in checks.items():
            for match in pattern.finditer(source):
                issues.append((str(relative), source.count("\n", 0, match.start()) + 1, kind))
        if path.suffix == ".py":
            try:
                tree = ast.parse(source, filename=str(relative))
                for item in tokenize.generate_tokens(io.StringIO(source).readline):
                    if item.type == tokenize.COMMENT and re.search(r"[\u4e00-\u9fff]", item.string):
                        issues.append((str(relative), item.start[0], "non-English comment"))
                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        doc = ast.get_docstring(node)
                        if doc and re.search(r"[\u4e00-\u9fff]", doc):
                            issues.append(
                                (str(relative), getattr(node, "lineno", 1), "non-English docstring")
                            )
            except (SyntaxError, tokenize.TokenError) as error:
                issues.append((str(relative), getattr(error, "lineno", 1), "syntax error"))
    for path, line, kind in issues:
        print(f"{path}:{line}: {kind}")
    print(f"Checked {count} source/documentation files; {len(issues)} issues.")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(check())
