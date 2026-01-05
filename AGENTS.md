# Overview
- We use Github for version control and CI.
- Code is written in python.

## Safety
- If you think that a command might be destructive, provide the command to the user to run. When in doubt ask the user.
- NEVER run `git checkout`.

## Python dev environment
- We use `uv` for venv management.
- Use python commands as `uv run python ...` and be sure that `pyproject.toml` is configured.

## Documentation standards
- Do not use excessive bold in markdown documents. Use font styling very selectively.
- Do not use emoji in either code or docs.
- Do not include "last updated" dates for documentation or code.
- Do not include copyright or author info in code.

## Code quality
- Maintain clean, readable code with clean separation of responsibilities and don't repeat yourself (DRY) design.
- Do not maintain legacy baggage. Remove old interfaces when refactoring. Do not keep thin wrappers.

## Linting
- Always run `ruff` for linting after significant work is completed.

```
uvx ruff check [path] --fix --extend-select I,B,SIM,C4,ISC,PIE
uvx ruff format [path]
```

- Fix any ruff errors manually as needed.

## Testing
- We write tests for all code and target >80% test coverage.
- Use `pytest` for test code.

## Miscellaneous
- When sharing commands for the user to run, always specify them as a single line. Do not split with ` on multiple lines.
