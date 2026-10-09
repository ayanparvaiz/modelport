import 'package:flutter/material.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:modelport_llamacpp/modelport_llamacpp.dart';

/// Only the manifest ships with the app; the GGUF file downloads on first use.
const bundle = 'asset://assets/manifests/smollm2-135m-instruct';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelPortFlutter.init(adapters: [LlamaCppAdapter()]);
  runApp(
    MaterialApp(
      title: 'ModelPort Chat',
      theme: ThemeData(colorSchemeSeed: Colors.indigo),
      home: const ChatPage(),
    ),
  );
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final _input = TextEditingController();
  final _messages = <ChatMessage>[];
  TextGenerator? _llm;
  String _status = 'Starting…';
  String _reply = '';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final llm = await TextGenerator.load(
        bundle,
        onProgress: (p) => setState(
          () =>
              _status = 'Downloading ${(p.fraction * 100).toStringAsFixed(0)}%',
        ),
      );
      setState(() {
        _llm = llm;
        _status = 'Ready: ${llm.model.manifest.name ?? llm.model.manifest.id}';
      });
    } on ModelPortException catch (error) {
      setState(() => _status = error.toString());
    }
  }

  Future<void> _send() async {
    final llm = _llm;
    final text = _input.text.trim();
    if (llm == null || text.isEmpty || _busy) return;
    _input.clear();
    setState(() {
      _messages.add(ChatMessage.user(text));
      _reply = '';
      _busy = true;
    });
    await for (final piece in llm.chat(_messages)) {
      setState(() => _reply += piece);
    }
    setState(() {
      _messages.add(ChatMessage.assistant(_reply.trim()));
      _reply = '';
      _busy = false;
    });
  }

  @override
  void dispose() {
    _llm?.close();
    _input.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('ModelPort · llama.cpp')),
    body: Column(
      children: [
        Padding(padding: const EdgeInsets.all(8), child: Text(_status)),
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(12),
            children: [
              for (final m in _messages)
                Align(
                  alignment: m.role == ChatRole.user
                      ? Alignment.centerRight
                      : Alignment.centerLeft,
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(10),
                      child: Text(m.content),
                    ),
                  ),
                ),
              if (_reply.isNotEmpty)
                Align(
                  alignment: Alignment.centerLeft,
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(10),
                      child: Text(_reply),
                    ),
                  ),
                ),
            ],
          ),
        ),
        SafeArea(
          child: Row(
            children: [
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(8),
                  child: TextField(
                    controller: _input,
                    onSubmitted: (_) => _send(),
                    decoration: const InputDecoration(
                      hintText: 'Ask something',
                    ),
                  ),
                ),
              ),
              IconButton(
                onPressed: _busy ? null : _send,
                icon: const Icon(Icons.send),
              ),
            ],
          ),
        ),
      ],
    ),
  );
}
