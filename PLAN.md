# ModelPort — Master Plan (A to Z)

> **Python diye model ready koro. Flutter e ek line e run koro.**
>
> Ei file ta amader puro roadmap. Ki banabo, keno banabo, kibhabe banabo, ar kon order e banabo, sob ekhane lekha.

| | |
|---|---|
| Plan version | v1 |
| Tarikh | 2026-10-09 |
| Lokkho | v0.1.0 free open source release + launch |
| Income / paid version | Ekhon na. Response bhalo hole pore alada plan hobe. |

**Kibhabe porbe:** VS Code e ei file khule `Cmd + Shift + V` chapo. Sundor preview dekhabe, chart gulo o thik moto dekhabe.

---

## Suchipotro

1. [Ek nojore](#1-ek-nojore)
2. [Problem ar solution](#2-problem-ar-solution)
3. [Ki ki banabo](#3-ki-ki-banabo)
4. [Big picture chart](#4-big-picture-chart)
5. [Developer journey: age vs pore](#5-developer-journey-age-vs-pore)
6. [App er bhitore model kibhabe chole](#6-app-er-bhitore-model-kibhabe-chole)
7. [Manifest spec: sobcheye important file](#7-manifest-spec-sobcheye-important-file)
8. [Python CLI details](#8-python-cli-details)
9. [Dart packages details](#9-dart-packages-details)
10. [Repo structure](#10-repo-structure)
11. [Tech stack ar tools](#11-tech-stack-ar-tools)
12. [Step by step kaj: Phase 0 theke 11](#12-step-by-step-kaj-phase-0-theke-11)
13. [Timeline chart ar milestone](#13-timeline-chart-ar-milestone)
14. [Testing plan](#14-testing-plan)
15. [Release process](#15-release-process)
16. [Risk ar tar solution](#16-risk-ar-tar-solution)
17. [v0.1 e ki thakbe NA](#17-v01-e-ki-thakbe-na)
18. [v0.1 er porer roadmap](#18-v01-er-porer-roadmap)
19. [Kaj korar niyom](#19-kaj-korar-niyom)
20. [Shekhar resource](#20-shekhar-resource)
21. [Shobdo kosh (glossary)](#21-shobdo-kosh-glossary)
22. [Ei shoptahe ki korbo](#22-ei-shoptahe-ki-korbo)

---

## 1. Ek nojore

| Ki | Uttor |
|---|---|
| Naam | **ModelPort** (proposed, Phase 0 te final) |
| Ek line e | PyTorch, Hugging Face ar GGUF model ke Flutter app e anar shobcheye shohoj rasta |
| Kader jonno | Flutter developer jara app e AI chay kintu ML jane na. ML developer jara nijer model app e dekhte chay. |
| Python part | `pip install modelport`. CLI diye export, verify, pack, publish, gen-dart. |
| Flutter part | `flutter pub add modelport modelport_onnx`. Ek API te sob dhoroner model. |
| License | Apache-2.0. Free, open source, company o use korte parbe. |
| Prothom target | v0.1.0, part-time kaj kore prai 14 shoptaho |

**Naam er obostha (2026-10-09 e check kora):**

| Jayga | Naam | Obostha |
|---|---|---|
| pub.dev | `modelport` | Free |
| PyPI | `modelport` | Free |
| GitHub org | `modelport` | Already onno keu niyeche. Amra `modelport-dev` nibo, eta free. |
| Hugging Face org | `modelport-dev` | Phase 0 te check korbo |

---

## 2. Problem ar solution

### Problem

Ajke ekjon Flutter developer nijer model ba Hugging Face er kono model app e dite chaile ei sob kaj nije korte hoy:

1. **Convert:** model ke mobile format e nite hoy (ONNX, `.pte`, GGUF). Kon format, kon setting, kew bole na.
2. **Shape khoja:** input er naam, shape, dtype khuje ber korte hoy. Onekei Netron khule guess kore.
3. **Preprocessing:** image resize, normalize, mean/std, NCHW layout Dart e hate likhte hoy. Ekta number bhul hole app crash kore na, sudhu bhul result dey. Eta sobcheye bipodjonok bug.
4. **Extra file:** labels file, tokenizer, chat template alada kore wire korte hoy.
5. **Download:** boro model app e bundle kora jay na. Download, resume, checksum, cache, update sob nije likhte hoy.
6. **Alada API:** prottek engine package er API alada. Engine bodlale code abar likhte hoy.
7. **Mil ache kina:** Python e je output ashe, phone e o sei output ashe kina, kew check kore na.

Engine package already ache: `flutter_onnxruntime`, `executorch_flutter`, `llm_llamacpp`. Kintu egulo sudhu model **run** kore. Upore lekha baki 6 ta kaj kono package e nai.

### Solution: ModelPort

- **Python CLI** convert, quantize, verify ar packaging kore. Sheshe ekta `modelport.json` manifest file banay.
- **Manifest** e model er sob tottho thake: input, output, preprocessing, labels, file URL, checksum, koto RAM lagbe.
- **Dart package** manifest pore nije nije download, verify, preprocess, run ar postprocess kore.
- **Adapter:** engine amra banabo na. Existing engine gulo ke ek API er niche anbo.

> **Mul niti:** Python model **ready** kore. Dart ar native engine model **run** kore. Phone er bhitore Python chalabo na, karon app size onek bere jay ar iOS e ML library pawa jay na.

---

## 3. Ki ki banabo

| # | Component | Bhasha | Kaj | Kothay publish |
|---|---|---|---|---|
| 1 | **Manifest Spec** | JSON Schema | Model er "ID card". Python likhe, Dart pore. | GitHub (`spec/`) |
| 2 | **modelport CLI** | Python | export, quantize, verify, pack, publish, gen-dart | PyPI |
| 3 | **modelport** (core) | Dart | manifest, download, cache, tensor, preprocessing, task API | pub.dev |
| 4 | **modelport_flutter** | Flutter | cache folder, asset loading, progress widget | pub.dev |
| 5 | **Adapters** | Dart | `modelport_onnx`, `modelport_executorch`, `modelport_llamacpp` | pub.dev |
| 6 | **Model Zoo** | JSON + HF | Ready, test kora model er list | GitHub + Hugging Face |
| 7 | **Demo app** | Flutter | Classify, Detect, Chat, Model manager screen | GitHub Releases (APK) |
| 8 | **Docs site** | Markdown | Guide, API reference, troubleshooting | GitHub Pages |

**Engine choice (2026-10-09 e pub.dev check kore):**

| Kaj | Engine package | Keno eta |
|---|---|---|
| Sadharon model (image, audio, text) | `flutter_onnxruntime` | Sobcheye active (Oct 2026 update), 6 platform, MIT. Purono `onnxruntime` package 2024 er por update hoy nai, tai ota na. |
| PyTorch native | `executorch_flutter` | Active (Sep 2026), 6 platform, MIT. Flutter 3.38+ lage. |
| LLM (GGUF) | `llm_llamacpp` | Active (Sep 2026), 5 platform, MIT, isolate e chole. Backup: `llama_cpp_dart`, beshi popular kintu Jan 2026 er por update nai. |
| Na nibo | `cactus` | pub.dev e license "unknown". Open source project e eta risk. |

---

## 4. Big picture chart

```
  [ ML DEVELOPER ER COMPUTER : Python ]

     PyTorch model      Hugging Face model      GGUF LLM
           │                    │                   │
           └────────────────────┼───────────────────┘
                                ▼
            ┌────────────────────────────────────────┐
            │         modelport CLI  (Python)        │
            │   export → quantize → verify → pack    │
            └───────────────────┬────────────────────┘
                                ▼
       Bundle = modelport.json + model file + labels + golden test
                                │
                                │  modelport publish
                                ▼
            ┌────────────────────────────────────────┐
            │    Hugging Face Hub  (free hosting)    │
            │    ba nijer server, ba app er asset    │
            └───────────────────┬────────────────────┘
                                │  prothom bar download, tarpor cache
                                ▼
  [ USER ER PHONE : Flutter app ]

         ImageClassifier.load('hf://modelport-dev/mobilenet')
                                │
            ┌───────────────────▼────────────────────┐
            │         modelport  (Dart core)         │
            │   manifest · cache · variant select    │
            │   preprocess · run · postprocess       │
            └───────┬─────────────┬─────────────┬────┘
                    ▼             ▼             ▼
             modelport_onnx  modelport_    modelport_       ← adapter: AMRA BANABO
                    │        executorch    llamacpp
                    ▼             ▼             ▼
             flutter_        executorch_   llm_llamacpp     ← engine: ALREADY ACHE
             onnxruntime     flutter
                    └─────────────┼─────────────┘
                                  ▼
                          CPU · GPU · NPU
```

**Ek kothay:** Python side e model ekbar ready hoy. Tarpor je kono Flutter app sei model ek line e use korte pare.

---

## 5. Developer journey: age vs pore

### Age (ajker obostha)

```
Model download ba train
  → Google kore khujo kon format lagbe
  → Convert script likho, error fix koro          (1 – 3 din)
  → Netron diye input shape guess koro
  → Dart e preprocessing hate likho               (bhul hole silent bug)
  → labels.txt asset e rakho
  → Download + cache code nije likho
  → Engine package er API shekho
  → Output bhul? Kothay bhul? → abar shuru theke

Somoy: prottek model e 3 – 7 din
```

### Pore (ModelPort diye)

Python side, ML developer er computer e:

```bash
pip install "modelport[onnx,hf]"
modelport export torchvision:mobilenet_v3_small --target onnx
modelport verify dist/mobilenet_v3_small
modelport publish dist/mobilenet_v3_small --hf modelport-dev/mobilenet_v3_small
```

Flutter side:

```bash
flutter pub add modelport modelport_flutter modelport_onnx
```

```dart
final classifier = await ImageClassifier.load('hf://modelport-dev/mobilenet_v3_small');
final results = await classifier.classify(imageBytes);
print(results.first);   // golden retriever (0.93)
```

**Target somoy: 15 – 30 minute.** Ar Zoo theke ready model nile Python part o lagbe na.

---

## 6. App er bhitore model kibhabe chole

`ImageClassifier.load(...)` call korle bhitore ei step gulo hoy:

```
 ImageClassifier.load('hf://modelport-dev/mobilenet_v3_small')
      │
      ▼
  1. Resolve      hf://  →  https://huggingface.co/.../resolve/main/modelport.json
      ▼
  2. Manifest     download kore pore, schema version check kore
      ▼
  3. Variant      kon adapter register kora? phone e koto RAM? → best variant bachai
      ▼
  4. Cache check  file ache ar sha256 thik? ──── ha ────► step 6
      │ na
      ▼
  5. Download     resume support · progress · sha256 verify · ".part" theke rename
      ▼
  6. Load         adapter.load(file)  →  engine session ready

 ────────────── protibar classify(image) call korle ──────────────

  7. Preprocess   decode → resize/crop → RGB → normalize → tensor   (manifest theke)
      ▼
  8. Run          adapter.run(tensor)
      ▼
  9. Postprocess  softmax → top-k → labels diye naam                 (manifest theke)
      ▼
 10. Result       [Classification('golden retriever', 0.93), ...]
```

Step 1 theke 6 ekbar hoy. Step 7 theke 10 protibar hoy.

---

## 7. Manifest spec: sobcheye important file

Manifest holo model er **ID card**. Model er sob gyan ei ek file e thake. Tai Dart code generic thakte pare. **Notun model mane notun manifest, notun code na.**

Ei spec bhalo hole onno developer ra o nijer engine er jonno adapter likhbe. Tokhon ModelPort ecosystem er center hoye jabe. Tai ei part e sobcheye beshi chinta korbo.

### Example 1: Image classification

```json
{
  "schema": "modelport/0.1",
  "id": "mobilenet_v3_small",
  "version": "1.0.0",
  "task": "image-classification",
  "license": "BSD-3-Clause",
  "source": "torchvision:mobilenet_v3_small",
  "variants": [
    {
      "id": "onnx-fp32",
      "runtime": "onnx",
      "precision": "fp32",
      "file": { "path": "onnx-fp32/model.onnx", "size": 10213441, "sha256": "9f2c..." },
      "min_ram_mb": 256
    },
    {
      "id": "executorch-xnnpack-fp32",
      "runtime": "executorch",
      "backend": "xnnpack",
      "precision": "fp32",
      "file": { "path": "executorch-fp32/model.pte", "size": 10400012, "sha256": "1ab4..." },
      "min_ram_mb": 256
    }
  ],
  "inputs": [
    {
      "name": "pixel_values",
      "dtype": "float32",
      "shape": [1, 3, 224, 224],
      "layout": "NCHW",
      "preprocess": {
        "type": "image",
        "resize": { "shorter_side": 256, "method": "bilinear", "antialias": false },
        "center_crop": [224, 224],
        "color": "RGB",
        "scale": 0.00392156862745098,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
      }
    }
  ],
  "outputs": [
    {
      "name": "logits",
      "dtype": "float32",
      "shape": [1, 1000],
      "postprocess": {
        "type": "classification",
        "activation": "softmax",
        "labels": { "path": "labels.txt", "size": 21675, "sha256": "c0d1..." },
        "top_k": 5
      }
    }
  ],
  "golden": {
    "inputs":  { "path": "golden/input_0.bin",  "size": 602112, "sha256": "..." },
    "outputs": { "path": "golden/output_0.bin", "size": 4000,   "sha256": "..." },
    "atol": 0.001,
    "rtol": 0.001
  }
}
```

### Example 2: LLM (chat model)

```json
{
  "schema": "modelport/0.1",
  "id": "qwen2.5-0.5b-instruct",
  "version": "1.0.0",
  "task": "text-generation",
  "license": "Apache-2.0",
  "source": "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF",
  "variants": [
    {
      "id": "gguf-q4_k_m",
      "runtime": "llamacpp",
      "precision": "q4_k_m",
      "file": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size": 491000000,
        "sha256": "..."
      },
      "min_ram_mb": 1024
    },
    {
      "id": "gguf-q8_0",
      "runtime": "llamacpp",
      "precision": "q8_0",
      "file": { "url": "https://huggingface.co/.../qwen2.5-0.5b-instruct-q8_0.gguf", "size": 676000000, "sha256": "..." },
      "min_ram_mb": 2048
    }
  ],
  "llm": {
    "context_length": 4096,
    "chat_template": "from_gguf",
    "defaults": { "temperature": 0.7, "top_p": 0.9, "max_tokens": 512 }
  }
}
```

Size ar sha256 er number gulo sudhu example. Asol number CLI nije boshabe.

### Field gulor mane

| Field | Mane | Lagbei? |
|---|---|---|
| `schema` | Spec er version, jemon `modelport/0.1` | Ha |
| `id`, `version` | Model er naam ar version (SemVer) | Ha |
| `task` | `image-classification`, `object-detection`, `text-generation`. Pore `embedding`, `speech-to-text`. | Ha |
| `license` | SPDX license naam, jemon `Apache-2.0` | Ha |
| `source` | Model kotha theke ashlo | Na |
| `variants[]` | Ek model er alada format ba precision | Ha, kompokkhe 1 ta |
| `variants[].runtime` | `onnx`, `executorch`, `llamacpp` | Ha |
| `variants[].file` | `path` ba `url`, sathe `size` ar `sha256` | Ha |
| `variants[].min_ram_mb` | Phone e er cheye kom RAM thakle ei variant bachai hobe na | Na |
| `inputs[]`, `outputs[]` | Tensor er naam, dtype, shape, layout | Tensor model e ha |
| `preprocess` | Chhobi ba data theke input tensor banano rule | Na |
| `postprocess` | Output tensor theke result banano rule | Na |
| `llm` | Context length, chat template, default setting | LLM e ha |
| `golden` | Test input, expected output, tolerance | Na, kintu Zoo te ha |

### Niyom gulo

1. **`path` holo relative.** Manifest jekhane ache, sekhan theke file khoja hoy. Tai ekoi bundle Hugging Face e, nijer server e, ba app er asset e rakha jay. Kothao URL bodlate hoy na.
2. **`url` holo absolute.** Boro LLM amra abar upload kori na. Original repo er link use kori. Eta license er jonno o bhalo.
3. **`sha256` ar `size` prottek file e lagbei.** Download er por mil na hole file fele dewa hoy.
4. **Manifest sudhu data.** Er moddhe kono code, script ba command thakbe na. Eta security er jonno.
5. **Resize er algorithm lekha lagbe.** Python er PIL ar Dart er `image` package ek bhabe resize kore na. Tai `method` ar `antialias` explicit lekha thake, ar dui jaygay ekoi algorithm implement hobe.
6. **Version er niyom:** notun optional field add korle minor version (`0.1` → `0.2`). Field remove ba mane bodlale major version. Dart notun major version dekhle clear error dibe.

---

## 8. Python CLI details

### Command gulo

| Command | Ki kore | Example |
|---|---|---|
| `doctor` | Python, torch, executorch version, disk space check kore | `modelport doctor` |
| `inspect` | `.onnx`, `.pte`, `.gguf` file er input, output, metadata dekhay | `modelport inspect model.onnx` |
| `export` | Model ke ek ba ekadhik format e convert kore, manifest banay | `modelport export hf:google/vit-base-patch16-224 --target onnx,executorch` |
| `quantize` | fp16 ba int8 variant banay | `modelport quantize dist/vit --int8` |
| `import-gguf` | Ready GGUF LLM theke manifest ar variant banay | `modelport import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF --quant q4_k_m,q8_0` |
| `verify` | Original ar exported model er output milay, golden test banay | `modelport verify dist/vit` |
| `pack` | sha256, size boshiye final bundle banay | `modelport pack dist/vit` |
| `publish` | Hugging Face e upload kore, `hf://` link print kore | `modelport publish dist/vit --hf modelport-dev/vit` |
| `gen-dart` | Manifest theke typed Dart class banay | `modelport gen-dart dist/vit/modelport.json -o lib/models/` |

### Model kotha theke nibe (`<source>`)

| Source | Mane | Example |
|---|---|---|
| `torchvision:` | torchvision er model | `torchvision:mobilenet_v3_small` |
| `hf:` | Hugging Face transformers model. Preprocessing nije nije `AutoImageProcessor` config theke ney. | `hf:google/vit-base-patch16-224` |
| `file:` | Nijer model. Ekta Python function je `(model, example_inputs)` return kore. | `file:my_model.py:build` |

> **Security:** Ochena `.pt` ba `.pth` file kokhono sadharon `torch.load` diye kholbo na. Pickle file er bhitore code chalano jay. `safetensors` ba `torch.load(..., weights_only=True)` use korbo.

### Export pipeline

```
                 ┌──────────────────────────┐
  <source>  ───► │  Loader                  │  torchvision / hf / file
                 └────────────┬─────────────┘
                              ▼
                 (model, example_inputs, preprocessing info)
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌───────────────┐    ┌──────────────┐
   │ ONNX        │    │ ExecuTorch    │    │ GGUF         │
   │ exporter    │    │ exporter      │    │ importer     │
   └──────┬──────┘    └───────┬───────┘    └──────┬───────┘
          └───────────────────┼───────────────────┘
                              ▼
                 verify  →  golden test  →  pack
                              ▼
                 dist/<id>/modelport.json + files
```

| Target | Kibhabe hobe |
|---|---|
| ONNX | `torch.onnx.export(..., dynamo=True)` → `onnx.checker` → chaile fp16 ba dynamic int8 (`onnxruntime.quantization`) |
| ExecuTorch | `torch.export.export` → `to_edge_transform_and_lower(..., partitioner=[XnnpackPartitioner()])` → `.to_executorch()` → `.pte` |
| GGUF (v0.1) | Ready GGUF file import. `gguf` pip package diye metadata pore. safetensors theke GGUF convert v0.2 te. |

Exporter gulo plugin er moto hobe, jate pore notun format add kora shohoj hoy:

```python
class Exporter(Protocol):
    target: str  # "onnx" | "executorch"

    def export(
        self,
        model: torch.nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        out_dir: Path,
        options: ExportOptions,
    ) -> list[Variant]: ...
```

### Verify kibhabe kaj kore

1. Ekoi input diye original PyTorch model ar exported model chalano hoy. Exported model chole Python er `onnxruntime` ba ExecuTorch runtime e.
2. Tulona hoy tin bhabe: max absolute difference, cosine similarity, ar classification hole top-1 mil.
3. Tolerance (`atol`, `rtol`) er moddhe thakle pass, na hole clear report diye fail.
4. Input ar expected output `golden/` folder e save hoy. Phone e o ei file diye test hoy. Eta amader **on-device parity test**, jeta ajke kono package dey na.

### Install er bhag (extras)

Base install halka thakbe. Je format dorkar sudhu setar library install hobe.

```toml
[project]
name = "modelport"
requires-python = ">=3.11"
dependencies = ["typer", "rich", "pydantic>=2", "jinja2", "huggingface_hub", "numpy", "pillow"]

[project.optional-dependencies]
onnx = ["torch", "onnx", "onnxscript", "onnxruntime"]
executorch = ["torch", "executorch"]
gguf = ["gguf"]
hf = ["transformers", "torchvision"]
all = ["modelport[onnx,executorch,gguf,hf]"]
```

`executorch` ekta nirdishto `torch` version chay. Tai docs e ekta "kon version er sathe kon version" table rakhbo.

---

## 9. Dart packages details

### Package er somporko

```
                       modelport   (pure Dart, Flutter chara o chole)
                           ▲
       ┌───────────────────┼─────────────────────┬──────────────────────┐
       │                   │                     │                      │
 modelport_flutter   modelport_onnx    modelport_executorch    modelport_llamacpp
                           │                     │                      │
                           ▼                     ▼                      ▼
                 flutter_onnxruntime    executorch_flutter         llm_llamacpp
```

Adapter alada package keno? Karon prottek engine app size baray. User sudhu je engine dorkar setai add korbe.

### App developer ja dekhbe (public API)

```dart
import 'package:flutter/widgets.dart';
import 'package:modelport/modelport.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:modelport_onnx/modelport_onnx.dart';
import 'package:modelport_llamacpp/modelport_llamacpp.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelPortFlutter.init(
    adapters: [OnnxAdapter(), LlamaCppAdapter()],
  );
  runApp(const MyApp());
}

// 1) Chhobi classify
final classifier = await ImageClassifier.load(
  'hf://modelport-dev/mobilenet_v3_small',
  onProgress: (p) => debugPrint('download ${(p.fraction * 100).round()}%'),
);
final results = await classifier.classify(imageBytes, topK: 3);

// 2) LLM chat, token by token
final llm = await TextGenerator.load('hf://modelport-dev/qwen2.5-0.5b-instruct');
await for (final piece in llm.chat([ChatMessage.user('Flutter ki?')])) {
  answer.write(piece);
}

// 3) Low-level: je kono custom model
final model = await ModelPort.load('asset://assets/models/my_model/modelport.json');
final outputs = await model.run({
  'input': Tensor.float32([1, 3, 224, 224], data),
});
await model.dispose();
```

### Bhitorer interface (adapter likhte ja lage)

```dart
/// Prottek engine er jonno ekta adapter.
abstract interface class RuntimeAdapter {
  /// 'onnx' | 'executorch' | 'llamacpp'
  String get runtime;

  /// Ei platform e ei variant chalano jabe kina.
  bool canRun(ModelVariant variant);
}

/// Tensor in, tensor out model (image, audio, sadharon model).
abstract interface class TensorAdapter implements RuntimeAdapter {
  Future<TensorSession> load(ResolvedVariant variant, Manifest manifest);
}

abstract interface class TensorSession {
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs);
  Future<void> dispose();
}

/// LLM er jonno alada, karon egulo tensor na, text stream dey.
abstract interface class TextGenerationAdapter implements RuntimeAdapter {
  Future<TextGenerationSession> load(ResolvedVariant variant, Manifest manifest);
}

abstract interface class TextGenerationSession {
  Stream<String> chat(List<ChatMessage> messages, {GenerationConfig? config});
  Future<void> cancel();
  Future<void> dispose();
}
```

### Core package er bhag

| Folder | Kaj |
|---|---|
| `manifest/` | JSON parse, validate, schema version check |
| `source/` | `hf://`, `https://`, `asset://`, `file://` resolve |
| `store/` | Download (resume, HTTP Range), sha256, cache folder, delete, cache size |
| `select/` | Variant bachai: kon adapter ache, kon platform, koto RAM |
| `tensor/` | `Tensor` class: dtype, shape, `Float32List` / `Int64List` data |
| `preprocess/` | Chhobi theke tensor (`package:image` diye) |
| `postprocess/` | softmax, top-k, labels, box decode, NMS |
| `tasks/` | `ImageClassifier`, `ObjectDetector`, `TextGenerator` |
| `errors/` | Clear error message, sathe fix er hint |

### Adapter gulor bishesh kotha

| Adapter | Ja mathay rakhte hobe |
|---|---|
| `modelport_onnx` | `flutter_onnxruntime` iOS ar macOS e input/output info dite pare na. Amra naam ar shape manifest theke nei. **Manifest er dorkar eikhanei proman hoy.** |
| `modelport_executorch` | `forward()` naam chara sudhu list ney. Manifest er `inputs` order onujayi list banabo. Flutter 3.38+ lagbe. |
| `modelport_llamacpp` | `llm_llamacpp` isolate e chole ar chat template GGUF theke ney. Amra manifest er default setting pass korbo. |

### Cache folder

```
<app support folder>/modelport/
└── mobilenet_v3_small/
    └── 1.0.0/
        ├── modelport.json
        ├── labels.txt
        └── onnx-fp32/
            └── model.onnx
```

### Error message hobe kajer

Library er code, comment ar error message English e hobe, karon user ra sara duniyar.

```
ModelPortException: No adapter registered for runtime "executorch".
Fix: add modelport_executorch to pubspec.yaml and pass ExecuTorchAdapter()
     to ModelPortFlutter.init(adapters: [...]).
```

---

## 10. Repo structure

Ek repo te sob thakbe (monorepo). Tai spec, Python ar Dart ek sathe bodlay, ar version mil thake.

```
modelport/
├── README.md
├── LICENSE                        Apache-2.0
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── DEVLOG.md                      protidin er choto note
├── IDEAS.md                       v0.1 er baire er idea
│
├── spec/
│   ├── manifest.schema.json       Pydantic theke auto banano
│   └── examples/                  Python ar Dart dui test ei file gulo use kore
│
├── python/                        PyPI: modelport
│   ├── pyproject.toml
│   ├── src/modelport/
│   │   ├── cli.py                 Typer app
│   │   ├── manifest/              Pydantic model + schema export
│   │   ├── sources/               torchvision, hf, file loader
│   │   ├── exporters/             onnx.py, executorch.py, gguf.py
│   │   ├── quantize/
│   │   ├── verify/                parity + golden
│   │   ├── inspect/
│   │   ├── codegen/               Jinja2 template → Dart
│   │   └── publish/               Hugging Face upload
│   └── tests/
│
├── dart/                          pub workspace
│   ├── pubspec.yaml               workspace root
│   └── packages/
│       ├── modelport/             core, pure Dart
│       ├── modelport_flutter/
│       ├── modelport_onnx/
│       ├── modelport_executorch/
│       └── modelport_llamacpp/
│
├── apps/
│   └── demo/                      Flutter demo app
│
├── zoo/
│   ├── index.json                 curated model list
│   └── models/<id>/modelport.json
│
├── docs/                          docs site (Zensical)
│
└── .github/
    ├── ISSUE_TEMPLATE/
    └── workflows/
        ├── python.yml
        ├── dart.yml
        ├── integration.yml
        ├── zoo.yml
        └── release.yml
```

---

## 11. Tech stack ar tools

| Jayga | Tool | Keno |
|---|---|---|
| Python | 3.12 + **uv** | Fast package ar virtual env manager |
| CLI | **Typer** + **Rich** | Shohoje sundor CLI, color, progress bar |
| Manifest | **Pydantic v2** | Validation, ar JSON Schema nije banay |
| Codegen | **Jinja2** | Dart file template |
| Python quality | **pytest**, **ruff**, **pyright** | Test, lint/format, type check |
| ML | torch, torchvision, transformers, onnx, onnxscript, onnxruntime, executorch, gguf, huggingface_hub | Export ar verify |
| Flutter | **3.38+** stable | `executorch_flutter` er native assets er jonno lagbe |
| Monorepo | **Dart pub workspaces** (melos optional) | Ek command e sob package |
| Dart lib | `http`, `crypto`, `image`, `path`, `path_provider` | Download, sha256, chhobi, file path |
| Dart test | `test`, `flutter_test`, `integration_test`, `mocktail` | Unit ar device test |
| Hosting | **Hugging Face Hub** | Public model free te host kora jay |
| Code | **GitHub**: Issues, Actions, Releases, Pages | Sob ek jaygay |
| Docs | **Zensical** | MkDocs Material team er notun tool. Material ekhon maintenance mode e. |
| Debug | **Netron** | Model er bhitor chokhe dekha |

**Device:** ekta Android phone (4 GB RAM er moto low-mid hole test er jonno bhalo), ekta iPhone ba simulator, ar Mac (iOS build er jonno lagbe).

**Install (macOS):**

```bash
brew install uv git gh
uv python install 3.12
flutter --version        # 3.38 ba tar upore hote hobe
flutter doctor           # sob green na hoya porjonto fix koro
```

---

## 12. Step by step kaj: Phase 0 theke 11

Prottek phase e thakbe **lokkho**, **kaj er checklist**, ar **kokhon shesh dhorbo**. Kaj shesh hole `[ ]` ke `[x]` kore dao.

---

### Phase 0: Setup ar 3 ta spike (Week 1 – 2)

**Lokkho:** Design er age nije chokhe dekha je 3 ta engine amar phone e asholei chole. Spike mane chhoto, fela dewar moto experiment. Ekhane problem dhorle pore 2 mash bachbe.

- [ ] Mac e Flutter 3.38+, Xcode, Android Studio, Python 3.12, uv install. `flutter doctor` sob green.
- [ ] GitHub e `modelport-dev` org khulo. Hugging Face e same naam e org khulo.
- [ ] PyPI account khulo, 2FA on koro. pub.dev er jonno Google account ready rakho.
- [ ] **Spike A:** torchvision `mobilenet_v3_small` → ONNX (choto Python script) → `flutter_onnxruntime` diye Android e ekta chhobi classify.
- [ ] **Spike B:** ekoi model → `.pte` (ExecuTorch, XNNPACK) → `executorch_flutter` diye run.
- [ ] **Spike C:** `Qwen2.5-0.5B-Instruct` GGUF (Q4_K_M) → `llm_llamacpp` diye chat.
- [ ] Prottek spike e note: app size koto barlo, load time, inference time, ki jhamela holo. Sob `notes/spikes.md` e.
- [ ] LLM engine final koro: `llm_llamacpp` naki `llama_cpp_dart`, spike er result dekhe.
- [ ] Naam final koro. Pub.dev, PyPI, GitHub abar check koro.

**Shesh jokhon:** 3 ta spike ekta real phone e chole, ar notes lekha ache.

---

### Phase 1: Manifest spec v0.1 (Week 2 – 3)

**Lokkho:** Python ar Dart er moddhe "contract" ta pakka kora.

- [ ] Monorepo banao: README, Apache-2.0 LICENSE, `.gitignore`, GitHub e push.
- [ ] `python/` e `uv init --package` diye project.
- [ ] Pydantic model likho: `Manifest`, `Variant`, `FileRef`, `TensorSpec`, `ImagePreprocess`, `ClassificationPostprocess`, `DetectionPostprocess`, `LlmConfig`.
- [ ] Pydantic theke JSON Schema export script → `spec/manifest.schema.json`.
- [ ] 3 ta example likho: classification, detection, text-generation → `spec/examples/`.
- [ ] Bhul example o likho (jemon sha256 nai). Test e egulo fail korte hobe.
- [ ] `docs/spec.md` e prottek field er explanation.
- [ ] Version er niyom docs e likho.

**Shesh jokhon:** Schema, example, test sob merge hoyeche, ar CI green.

---

### Phase 2: Python CLI ar ONNX export (Week 3 – 5)

**Lokkho:** Prothom kajer tool. Ek command e model theke verified bundle.

- [ ] Typer app skeleton. `modelport --help` e sob command dekhay.
- [ ] `doctor` command.
- [ ] `inspect` command, `.onnx` file er jonno.
- [ ] Source loader: `torchvision:`, `hf:` (image classification), `file:script.py:fn`.
- [ ] `hf:` model er preprocessing `AutoImageProcessor` config theke auto nibe: size, mean, std, resample.
- [ ] ONNX exporter: `dynamo=True`, `onnx.checker`, opset thik kora.
- [ ] `quantize`: fp16 ar dynamic int8 variant.
- [ ] `verify`: PyTorch vs onnxruntime tulona, golden file save.
- [ ] `pack`: sha256, size, final manifest.
- [ ] `publish --hf`: `huggingface_hub` diye upload.
- [ ] Unit test, tiny model diye jate fast chole.
- [ ] GitHub Actions `python.yml`: ruff, pyright, pytest.

**Shesh jokhon:** Ei tin ta command kaj kore, ar Hugging Face e bundle dekha jay:

```bash
modelport export torchvision:mobilenet_v3_small --target onnx
modelport verify dist/mobilenet_v3_small
modelport publish dist/mobilenet_v3_small --hf modelport-dev/mobilenet_v3_small
```

---

### Phase 3: Dart core (Week 5 – 7)

**Lokkho:** Engine chara puro flow Dart e kaj kore.

- [ ] `dart/` e pub workspace. `packages/modelport` banao (`dart create -t package`).
- [ ] Manifest class, `fromJson`, validation. `spec/examples` diye test, tai Python ar Dart ekoi file e test hoy.
- [ ] `Tensor` class: dtype, shape, data.
- [ ] Source resolver: `hf://`, `https://`, `asset://`, `file://`.
- [ ] `ModelStore`: progress stream, resume (HTTP Range), sha256, `.part` theke rename, cancel, delete, cache size.
- [ ] `VariantSelector`: registered adapter, platform, RAM dekhe bachai.
- [ ] Image preprocessing: decode, EXIF rotation thik, resize (manifest er algorithm), crop, normalize, NCHW ba NHWC.
- [ ] Postprocessing: softmax, top-k, labels.
- [ ] **Cross-language test:** Python e preprocess kora tensor ar Dart e preprocess kora tensor er difference choto kina.
- [ ] `RuntimeAdapter` interface, ar test er jonno ekta fake adapter.
- [ ] `dart.yml` CI: format, analyze, test.
- [ ] `packages/modelport_flutter`: `path_provider` diye cache folder, asset loader, `ModelPortFlutter.init`.

**Shesh jokhon:** Fake adapter diye puro flow (resolve → download → verify → preprocess → run → postprocess) unit test e pass kore.

---

### Phase 4: ONNX adapter ar "First Light" (Week 7)

**Lokkho:** Prothom bar asol phone e asol model, ModelPort diye.

- [ ] `packages/modelport_onnx`: `flutter_onnxruntime` wrap. `Tensor` ↔ `OrtValue` convert. Thread setting.
- [ ] `ImageClassifier` task API.
- [ ] Chhoto example app: gallery theke chhobi → top-3 result.
- [ ] **On-device golden test:** `integration_test` e golden input chalao, expected output er sathe milao.
- [ ] Android ar iOS dui jaygay chalao.

**Shesh jokhon:** Phone e `ImageClassifier.load('hf://modelport-dev/mobilenet_v3_small')` thik result dey.

> **Eta prothom boro milestone.** Ekta 20 second er video record koro ar post koro. Eta "build in public" er shuru.

---

### Phase 5: ExecuTorch (Week 8)

**Lokkho:** Ek bundle, dui engine, same result.

- [ ] Python: ExecuTorch exporter (`torch.export` → XNNPACK → `.pte`).
- [ ] `inspect` e `.pte` support.
- [ ] `verify`: PyTorch vs ExecuTorch Python runtime.
- [ ] `packages/modelport_executorch`: `ExecuTorchModel.load(path)`, `forward(list)`, input order manifest theke.
- [ ] Mobilenet bundle e dui variant: onnx ar executorch.
- [ ] Dui engine er tulona: speed, app size, top-1 same kina. Result docs e table akare.

**Shesh jokhon:** Ekoi app e engine bodlale result same thake.

---

### Phase 6: LLM, GGUF (Week 9)

**Lokkho:** Phone e offline chat, ek line e.

- [ ] Python `import-gguf`: HF GGUF repo er file list, quant bachai, `gguf` diye metadata (context length, architecture), sha256.
- [ ] Manifest e ekadhik variant (q4_k_m, q8_0) ar `min_ram_mb`.
- [ ] `packages/modelport_llamacpp`: `llm_llamacpp` wrap, streaming chat, stop/cancel.
- [ ] `TextGenerator` task API: `chat()`, `generate()`, `GenerationConfig`.
- [ ] Phone er RAM pora (Android ar iOS), jate kom RAM e choto variant nay.
- [ ] Boro file test: 400 MB+ download er majhe net bondho koro. Net ashle resume hote hobe.

**Shesh jokhon:** Qwen2.5-0.5B phone e stream kore uttor dey, ar download resume kaj kore.

---

### Phase 7: Object detection ar gen-dart (Week 10)

**Lokkho:** Dwitiyo task, ar custom model er jonno typed code.

- [ ] Detection model bachai, **license dekhe**. YOLOX-Nano (Apache-2.0) ba torchvision SSDLite (BSD-3). Spike kore je ta shohoje export hoy.
- [ ] **Ultralytics YOLOv8/YOLO11 na.** Egulo AGPL-3.0, amader Apache project ar Zoo te jhamela korbe.
- [ ] Postprocess: box decode, NMS, score threshold. Setting manifest e.
- [ ] `ObjectDetector` task API. Box er coordinate original chhobir size e.
- [ ] `gen-dart`: Jinja2 template theke typed class (input/output naam, shape constant, `predict` method).
- [ ] CI te generated code `dart analyze` pass kore kina test.

**Shesh jokhon:** Detection demo kaj kore, ar custom model generated class diye chalano jay.

---

### Phase 8: Demo app ar Model Zoo (Week 11)

**Lokkho:** Keu code na likhe o ModelPort er shokti dekhte pare.

- [ ] `apps/demo` screen gulo:
  - Home
  - Classify (camera ba gallery)
  - Detect (chhobir upor box)
  - Chat (LLM)
  - Models (download, delete, koto jayga nicche)
  - Bench (load time, latency)
- [ ] Classify screen e engine switch: ONNX ↔ ExecuTorch.
- [ ] Zoo: `zoo/index.json` ar 5 ta model, sob Apache, MIT ba BSD license er:
  - `mobilenet_v3_small` (classification)
  - `efficientnet_b0` (classification)
  - `yolox_nano` ba `ssdlite` (detection)
  - `qwen2.5-0.5b-instruct` (LLM)
  - `smollm2-360m-instruct` (LLM)
- [ ] `zoo.yml` CI: prottek manifest schema valid, URL kaj kore, sha256 mile.
- [ ] Demo APK GitHub Releases e.

**Shesh jokhon:** Keu APK install kore 5 ta model try korte pare.

---

### Phase 9: Docs (Week 12)

**Lokkho:** Docs pore notun keu nije nije chalate pare.

- [ ] README: ek line pitch, GIF, 3 step quickstart, engine table, link.
- [ ] Docs site (Zensical, GitHub Pages):
  - Getting started: Flutter developer er jonno
  - Getting started: ML developer er jonno
  - Concepts: manifest, variant, adapter
  - Guide: nijer PyTorch model, HF model, GGUF LLM, offline asset e model bundle
  - CLI reference
  - Troubleshooting: iOS minimum version, ProGuard rule, 16 KB page size, app size
  - Adapter likhar guide, contributor der jonno
- [ ] Sob public Dart API te dartdoc comment. pub.dev nije API docs banay.
- [ ] Prottek package e README, CHANGELOG, `example/`.
- [ ] **Fresh eyes test:** ekjon bondhu ke sudhu docs diye 30 minute e chalate bolo. Kothay atke, note koro ar thik koro.

**Shesh jokhon:** Fresh eyes test pass.

---

### Phase 10: Quality, CI ar release v0.1.0 (Week 13)

**Lokkho:** Clean machine e install kore kaj kore.

- [ ] `integration.yml`: Android emulator e demo app er golden test. macOS runner e iOS simulator.
- [ ] `pana` diye pub.dev score check, sob warning thik.
- [ ] Security check: sudhu https, sha256 lagbei, manifest e code nai, Python e unsafe pickle load nai.
- [ ] Device test table: low-end Android (4 GB ba kom), mid Android, iPhone.
- [ ] PyPI trusted publishing setup. pub.dev automated publishing (GitHub Actions, tag diye).
- [ ] `0.1.0` tag → PyPI, pub.dev (5 package), GitHub Release.
- [ ] Notun machine e test: `pip install modelport`, notun Flutter project e `flutter pub add`.

**Shesh jokhon:** Sob jaygay 0.1.0 live, ar clean install kaj kore.

---

### Phase 11: Launch (Week 14)

**Lokkho:** Manush jane, try kore, feedback dey.

- [ ] Blog post (dev.to, Medium ba Hashnode): "Your PyTorch model in a Flutter app in 3 commands".
- [ ] 60 – 90 second demo video: YouTube, X, LinkedIn.
- [ ] Post: r/FlutterDev, r/LocalLLaMA (LLM er angle), Hacker News "Show HN", Flutter Discord, Hugging Face community.
- [ ] fluttergems.dev ar awesome-flutter list e submit.
- [ ] Bangladesh community: GDG, Flutter Dhaka, Facebook dev group e Bangla post.
- [ ] Launch er por 2 shoptaho: issue te 24 – 48 ghontar moddhe reply. Beshi chawa feature gulo note koro.

**Shesh jokhon:** Launch hoyeche, ar feedback list theke v0.2 plan ready.

---

## 13. Timeline chart ar milestone

Part-time, dine 2 – 3 ghonta dhore hishab. Deri hole problem nai. **Order ta important, tarikh na.**

```
Week                      1   2   3   4   5   6   7   8   9   10  11  12  13  14
                          ───────────────────────────────────────────────────────
P0  Setup + 3 spike       ███ ███ ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
P1  Manifest spec         ·   ███ ███ ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·
P2  Python CLI + ONNX     ·   ·   ███ ███ ███ ·   ·   ·   ·   ·   ·   ·   ·   ·
P3  Dart core             ·   ·   ·   ·   ███ ███ ███ ·   ·   ·   ·   ·   ·   ·
P4  ONNX + First Light    ·   ·   ·   ·   ·   ·   ███ ·   ·   ·   ·   ·   ·   ·
P5  ExecuTorch            ·   ·   ·   ·   ·   ·   ·   ███ ·   ·   ·   ·   ·   ·
P6  LLM (GGUF)            ·   ·   ·   ·   ·   ·   ·   ·   ███ ·   ·   ·   ·   ·
P7  Detection + gen-dart  ·   ·   ·   ·   ·   ·   ·   ·   ·   ███ ·   ·   ·   ·
P8  Demo app + Zoo        ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ███ ·   ·   ·
P9  Docs                  ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ███ ·   ·
P10 Release v0.1.0        ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ███ ·
P11 Launch                ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ███
```

| Milestone | Kokhon | Ki proman hobe |
|---|---|---|
| **M1** Spike done | Week 2 shesh | 3 ta engine phone e chole |
| **M2** CLI works | Week 5 shesh | Ek command e verified ONNX bundle Hugging Face e |
| **M3** First Light | Week 7 shesh | Phone e ek line e classify |
| **M4** LLM chat | Week 9 shesh | Offline chat, download resume |
| **M5** Demo ready | Week 11 shesh | APK te 5 ta model |
| **M6** v0.1.0 | Week 13 shesh | PyPI ar pub.dev e live |
| **M7** Launch | Week 14 | Public announcement |

---

## 14. Testing plan

```
                      /\
                     /  \       Manual device test
                    /    \      low-end Android, mid Android, iPhone
                   /──────\
                  /        \    Integration test
                 /          \   emulator/simulator e on-device golden test
                /────────────\
               /              \   Cross-language test
              /                \  Python vs Dart, shared spec/examples
             /──────────────────\
            /                    \   Unit test
           /                      \  pytest ar dart test. Sobcheye beshi eita.
          /────────────────────────\
```

| Layer | Ki test hoy | Tool | Kokhon chole |
|---|---|---|---|
| Unit (Python) | Manifest, exporter (tiny model), codegen snapshot | pytest | Prottek push e |
| Unit (Dart) | Manifest parse, store, selector, pre/postprocess | `dart test` | Prottek push e |
| Cross-language | Ekoi chhobi Python ar Dart e preprocess kore mil ache kina | pytest + `dart test` | Prottek push e |
| Export (slow) | Asol model export ar verify | pytest `-m slow` | Raat e ekbar (nightly) |
| Integration | Demo app e golden test | `integration_test` | PR e ar release er age |
| Manual | Real phone e speed, RAM, battery | Demo app er Bench screen | Release er age |

**Golden test er idea:** Python e model er output save kore rakhi. Phone e same input diye chalai. Dui output tolerance er moddhe thakle pass. Eta dhore preprocessing ba conversion er silent bug.

---

## 15. Release process

**Version:** SemVer. `0.x` mane API ekhono bodlate pare. Python ar Dart package er version alada, kintu manifest `schema` version dujonei mane.

**Release checklist:**

1. [ ] CHANGELOG update.
2. [ ] Version bump: `pyproject.toml` ar prottek `pubspec.yaml`.
3. [ ] `dart pub publish --dry-run` sob package e. Warning thakle thik koro.
4. [ ] PR → CI green → merge.
5. [ ] Tag dao:
   - `py-v0.1.0` → GitHub Actions PyPI te publish kore
   - `modelport-v0.1.0`, `modelport_flutter-v0.1.0`, `modelport_onnx-v0.1.0` ... → pub.dev e publish
6. [ ] GitHub Release note ar demo APK.
7. [ ] Announce.

**Publish er order:** age `modelport` (core), tarpor `modelport_flutter`, tarpor adapter gulo. Karon egulo core er published version er upor nirbhor kore.

> **Sabdhan:** pub.dev e publish kora version muche fela jay na. Tai sobsomoy age `--dry-run`.

---

## 16. Risk ar tar solution

| Risk | Ki hote pare | Solution |
|---|---|---|
| Engine package bondho hoye gelo | Adapter bhenge jabe | Adapter alada package, version pin, dorkar hole fork. Bhobishyote nijer FFI. |
| Preprocessing mil nai | Silent bhul result | Manifest e resize algorithm explicit, cross-language ar golden test |
| App size boro hoye jay | User der complain | Alada adapter package. Docs e prottek adapter er size table. |
| Boro download fail kore | LLM download hoy na | Resume, sha256, Wi-Fi only option |
| Kom RAM e crash | LLM e app bondho | `min_ram_mb`, variant selector, age theke warning |
| Model license jhamela | Legal problem | Zoo te sudhu Apache, MIT, BSD. `license` field lagbei. Demo te license dekhabe. |
| iOS build jhamela | iOS user atke jay | macOS CI, docs e iOS troubleshooting |
| Python version jhamela | torch ar executorch mil na | Extras, pinned version table |
| Scope beshi hoye jawa | Kokhono release hoy na | Section 17 er "na" list. Notun idea `IDEAS.md` e. |
| Klanti, burnout | Project majhpothe bondho | Choto choto milestone, protisoptahe demo, build in public |

---

## 17. v0.1 e ki thakbe NA

Ei list ta scope er pahara. Notun idea ashle **`IDEAS.md` e likho, v0.1 e dhukaio na.**

- ❌ Web support
- ❌ Training ba fine-tuning
- ❌ Cloud API (OpenAI, Gemini ityadi). Egular jonno already onek package ache.
- ❌ Nijer native engine ba C++ code
- ❌ TFLite/LiteRT, speech, vision LLM, embedding, RAG
- ❌ Server, account, paid feature

---

## 18. v0.1 er porer roadmap

| Version | Kokhon (mota dag) | Ki thakbe |
|---|---|---|
| **v0.2** | Launch er 6 shoptaho por | Embedding + tokenizer, speech-to-text (`sherpa_onnx` adapter), TFLite/LiteRT adapter, ExecuTorch int8, safetensors theke GGUF convert, background download, camera stream helper |
| **v0.3** | Tar 2 mash por | Web support, hardware acceleration (CoreML, Metal, NPU), vision LLM, system model adapter (iOS er Apple Foundation Models, Android er Gemini Nano), existing RAG package er sathe integration |
| **v1.0** | Stable hole | Manifest spec v1, API freeze, long-term support |
| **Advanced / paid** | Response bhalo hole | Ekhon kono kaj na. Pore alada plan. |

v0.2 er asol order thik korbe launch er feedback. Upore ja ache seta sudhu dharona.

---

## 19. Kaj korar niyom

**Git:**
- `main` branch protected. Kaj hobe `feat/...`, `fix/...` branch e.
- Eka holeo PR khulo. CI green hole merge. Eta pore contributor der jonno abhyash.
- Commit message: Conventional Commits. `feat:`, `fix:`, `docs:`, `test:`, `chore:`.

**Kaj er hishab:**
- GitHub Projects board: `Backlog → This week → Doing → Done`.
- Prottek phase er checklist ke GitHub issue banao.

**Protidin (2 – 3 ghonta):**
- Shurute 5 minute: `DEVLOG.md` e likho "aj ki korbo".
- Sheshe 5 minute: likho "ki holo, kothay atkalam".

**Protisoptaho:**
- Shukrobar: ekta demo, screenshot ba video. X ba LinkedIn e post (build in public). Launch er age i manush jante shuru korbe.
- Robibar: porer shoptaher 3 ta main kaj bachai.

**Atke gele:**
- 1 ghontar beshi atkale note kore onno kaj e jao, pore fire asho.
- Engine package er GitHub Discussions ba Flutter Discord e proshno koro.

**Bhasha:** Code, comment, docs, error message English e. Nijer note ar DEVLOG Bangla te cholbe.

---

## 20. Shekhar resource

| Bishoy | Link |
|---|---|
| ExecuTorch docs | https://pytorch.org/executorch |
| PyTorch ONNX export | https://docs.pytorch.org/docs/stable/onnx.html |
| ONNX Runtime quantization | https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html |
| llama.cpp (GGUF) | https://github.com/ggml-org/llama.cpp |
| Hugging Face Hub (Python) | https://huggingface.co/docs/huggingface_hub |
| flutter_onnxruntime | https://pub.dev/packages/flutter_onnxruntime |
| executorch_flutter | https://pub.dev/packages/executorch_flutter |
| llm_llamacpp | https://pub.dev/packages/llm_llamacpp |
| Dart pub workspaces | https://dart.dev/tools/pub/workspaces |
| Dart native assets ar FFI | https://dart.dev/interop/c-interop |
| pub.dev e publish | https://dart.dev/tools/pub/publishing |
| pub.dev automated publishing | https://dart.dev/tools/pub/automated-publishing |
| PyPI trusted publishing | https://docs.pypi.org/trusted-publishers/ |
| uv | https://docs.astral.sh/uv/ |
| Typer | https://typer.tiangolo.com |
| Pydantic | https://docs.pydantic.dev |
| Zensical | https://zensical.org |
| Netron | https://netron.app |

---

## 21. Shobdo kosh (glossary)

| Shobdo | Mane |
|---|---|
| **Manifest** | `modelport.json`. Model er ID card, sob tottho ek file e. |
| **Variant** | Ekoi model er alada rup. Jemon ONNX fp32, ExecuTorch, GGUF Q4. |
| **Adapter** | Engine package ke ModelPort er API te jora deyar Dart package. |
| **Engine / Runtime** | Je library asholei model chalay. Jemon ONNX Runtime, ExecuTorch, llama.cpp. |
| **Quantization** | Model er number choto kore (fp32 theke int8 ba Q4) size ar RAM komano. |
| **Preprocessing** | Chhobi ba text ke model er input tensor e rupantor. |
| **Postprocessing** | Model er output ke manush er bujhar moto result e rupantor. |
| **Golden test** | Python er saved output er sathe phone er output milano. |
| **Parity** | Dui jaygay output same thaka. |
| **Spike** | Chhoto, fela dewar moto experiment, jhuki age bhag dhorar jonno. |
| **Zoo** | Ready, test kora model er list. |
| **Bundle** | Manifest + model file + labels + golden, ek folder e. |

---

## 22. Ei shoptahe ki korbo

Ajkei shuru korar 5 ta kaj:

1. [ ] `flutter --version` dekho. 3.38 er niche hole `flutter upgrade`.
2. [ ] `brew install uv` ar `uv python install 3.12`.
3. [ ] GitHub e `modelport-dev` org ar Hugging Face e same naam e org khulo.
4. [ ] `notes/spikes.md` file banao, ar **Spike A** shuru koro: mobilenet → ONNX → Android e classify.
5. [ ] `DEVLOG.md` e prothom line likho: "Day 1: ModelPort shuru."

> **Mone rakho:** Prothom version perfect howar dorkar nai. Kaj kore, docs ache, ar manush try korte pare, eitukui jothesto. Baki sob feedback theke ashbe.
