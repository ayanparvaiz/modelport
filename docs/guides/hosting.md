# Hosting bundles

A bundle can live anywhere an app can reach.

| Where | Location string | How to publish |
|---|---|---|
| GitHub release | `https://github.com/you/repo/releases/download/tag/<id>.json` | `modelport publish <bundle> --github you/repo --tag tag` |
| Hugging Face Hub | `hf://you/name` or `hf://you/name@revision` | `modelport publish <bundle> --hf you/name` |
| Any HTTPS server | `https://example.com/models/<id>/` | Copy the bundle folder |
| Flutter assets | `asset://assets/models/<id>` | Add the folders to `pubspec.yaml` |
| A local folder | `/absolute/path/to/bundle` | Nothing to do |

Plain `http://` is refused, except for `localhost`, which is handy during development with `adb reverse`.

## Flutter assets

List every folder of the bundle in `pubspec.yaml`, because Flutter asset folders are not recursive:

```yaml
flutter:
  assets:
    - assets/models/mobilenet_v3_small/
    - assets/models/mobilenet_v3_small/onnx-fp32/
    - assets/models/mobilenet_v3_small/golden/
```

Files are copied into the cache and verified on first use. You can also ship only `modelport.json` as an asset and let the weights download, as the llama.cpp example does.
