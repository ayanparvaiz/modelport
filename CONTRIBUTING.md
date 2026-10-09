# Contributing to ModelPort

Thanks for your interest in ModelPort. This project is in early development, so the best contributions right now are bug reports, design feedback on the manifest spec, and small focused pull requests.

## Ground rules

- Be kind. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- Open an issue before starting large changes, so we can agree on the approach first.
- Keep pull requests small. One logical change per pull request is easier to review.

## Repository layout

| Path | Language | Notes |
|---|---|---|
| `spec/` | JSON Schema | The manifest contract shared by Python and Dart |
| `python/` | Python 3.11+ | The `modelport` CLI, managed with [uv](https://docs.astral.sh/uv/) |
| `dart/` | Dart 3 / Flutter | A pub workspace with all Dart packages |
| `apps/demo/` | Flutter | Demo app |
| `zoo/` | JSON | Curated model manifests |

## Development setup

### Python CLI

```bash
cd python
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

### Dart packages

```bash
cd dart
dart pub get
dart test packages/modelport
```

You need Flutter 3.38 or newer for the Flutter packages.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use it for |
|---|---|
| `feat:` | A new feature |
| `fix:` | A bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or fixing tests |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `chore:` | Tooling, CI, dependencies |

Add a scope when it helps, for example `feat(cli): add inspect command` or `fix(dart): handle empty labels file`.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your change and add tests.
3. Make sure all checks pass locally.
4. Open a pull request that explains what changed and why.

## Changing the manifest spec

The manifest is a contract between the Python CLI and every Dart package. Changes to `spec/` need:

- an update to the Pydantic models in `python/src/modelport/manifest/`
- a regenerated `spec/manifest.schema.json`
- matching changes to the Dart parser
- an example in `spec/examples/` that exercises the change

## Adding a model to the zoo

Only models with permissive licenses (Apache-2.0, MIT, BSD) are accepted in the zoo. Every zoo manifest must include `sha256` for every file and a `golden` test that passes.
