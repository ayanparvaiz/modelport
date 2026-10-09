/// Models from the ModelPort zoo, hosted as GitHub release assets.
const zooBase =
    'https://github.com/ayanparvaiz/modelport/releases/download/zoo-v1';

class ZooModel {
  const ZooModel(this.id, this.title, this.note);

  final String id;
  final String title;
  final String note;

  String get location => '$zooBase/$id.json';
}

/// An engine choice, expressed as the bundle variant it loads.
class Engine {
  const Engine(this.label, this.variantId);

  final String label;
  final String variantId;
}

const classifiers = [
  ZooModel('mobilenet_v3_small', 'MobileNetV3 Small', '10 MB, fast CNN'),
  ZooModel('deit-tiny-patch16-224', 'DeiT Tiny', '24 MB, vision transformer'),
];

const detector = ZooModel('yolos-tiny', 'YOLOS Tiny', '27 MB, 80 COCO objects');

const chatModels = [
  ZooModel('smollm2-135m-instruct', 'SmolLM2 135M', '105 MB download'),
  ZooModel('qwen2.5-0.5b-instruct', 'Qwen2.5 0.5B', '491 MB download'),
];

const classifierEngines = [
  Engine('ONNX', 'onnx-fp32'),
  Engine('ONNX int8', 'onnx-int8'),
  Engine('ExecuTorch', 'executorch-xnnpack-fp32'),
];

const detectorEngines = [
  Engine('ONNX', 'onnx-fp32'),
  Engine('ExecuTorch', 'executorch-xnnpack-fp32'),
];
