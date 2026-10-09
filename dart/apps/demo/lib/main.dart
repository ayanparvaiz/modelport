import 'package:flutter/material.dart';
import 'package:modelport_executorch/modelport_executorch.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:modelport_llamacpp/modelport_llamacpp.dart';
import 'package:modelport_onnx/modelport_onnx.dart';

import 'pages/chat_page.dart';
import 'pages/classify_page.dart';
import 'pages/detect_page.dart';
import 'pages/models_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelPortFlutter.init(
    adapters: [OnnxAdapter(), ExecuTorchAdapter(), LlamaCppAdapter()],
  );
  runApp(const DemoApp());
}

class DemoApp extends StatelessWidget {
  const DemoApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'ModelPort',
    theme: ThemeData(colorSchemeSeed: Colors.teal),
    darkTheme: ThemeData(
      colorSchemeSeed: Colors.teal,
      brightness: Brightness.dark,
    ),
    home: const HomePage(),
  );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _tab = 0;

  static const _titles = ['Classify', 'Detect', 'Chat', 'Models'];

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: Text('ModelPort · ${_titles[_tab]}')),
    body: IndexedStack(
      index: _tab,
      children: const [ClassifyPage(), DetectPage(), ChatPage(), ModelsPage()],
    ),
    bottomNavigationBar: NavigationBar(
      selectedIndex: _tab,
      onDestinationSelected: (i) => setState(() => _tab = i),
      destinations: const [
        NavigationDestination(
          icon: Icon(Icons.image_search),
          label: 'Classify',
        ),
        NavigationDestination(
          icon: Icon(Icons.center_focus_strong),
          label: 'Detect',
        ),
        NavigationDestination(
          icon: Icon(Icons.chat_bubble_outline),
          label: 'Chat',
        ),
        NavigationDestination(icon: Icon(Icons.storage), label: 'Models'),
      ],
    ),
  );
}
