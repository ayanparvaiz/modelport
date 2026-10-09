# Release 0.1.0: tomar kaj

Sob kichu ready. Nicher step gulo tomar account lage, tai tumi nije korbe. Mot 10–15 minute.

## 1. pub.dev: 5 ta package (ei order e i)

Core age, karon baki gulo oitar upor nirbhor kore. Prottekta te dry-run e 0 warning chilo.

```bash
cd ~/Documents/UniqueSkills/dart/packages/modelport && dart pub publish
cd ../modelport_flutter && flutter pub publish
cd ../modelport_onnx && flutter pub publish
cd ../modelport_executorch && flutter pub publish
cd ../modelport_llamacpp && dart pub publish
```

Prottek bar `y` diye confirm korte hobe. Publish kora version ar muche fela jay na.

Publish er 10–20 minute por pub.dev score ashbe. Core package local e 160/160 peyeche.

**Porer version gulo automatic korte (optional):** prottek package er pub.dev page e Admin tab → "Publishing from GitHub Actions" enable → repository `ayanparvaiz/modelport`, tag pattern `<package>-v{{version}}` (jemon `modelport_onnx-v{{version}}`). Tarpor `git tag modelport_onnx-v0.1.1 && git push --tags` dilei publish hobe.

## 2. PyPI: Python CLI

Ekbar setup (pypi.org e login kore):

1. Account settings → Publishing → **Add a new pending publisher**
2. PyPI project name: `modelport`
3. Owner: `ayanparvaiz`, Repository: `modelport`
4. Workflow name: `release-python.yml`, Environment name: `pypi`

Tarpor terminal e:

```bash
cd ~/Documents/UniqueSkills
git tag py-v0.1.0
git push origin py-v0.1.0
```

GitHub Actions nijei build kore PyPI te publish korbe. Kono token lage na.

## 3. GitHub release

https://github.com/ayanparvaiz/modelport/releases e `ModelPort 0.1.0` draft ache, demo APK (65 MB, arm64) shoho. "Edit" → tag `v0.1.0` → **Publish release**.

## 4. Launch

`notes/launch/README.md` e order ar sob post er draft ache.
