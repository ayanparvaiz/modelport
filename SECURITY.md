# Security Policy

## Supported versions

ModelPort is in early development. Security fixes are applied to the latest release only.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/ayanparvaiz/modelport/security/advisories/new). You will get a response within 7 days.

## Security design principles

- **Manifests are data only.** A `modelport.json` file never contains code, scripts, or commands.
- **Every file is verified.** Each downloaded file must match the `sha256` and `size` in its manifest, or it is deleted.
- **HTTPS only.** Remote model files are downloaded over HTTPS.
- **No unsafe pickle loading.** The CLI never calls `torch.load` on untrusted checkpoints without `weights_only=True`. Prefer `safetensors`.
