import 'dart:async';

import 'package:flutter/material.dart';
import 'package:modelport_flutter/modelport_flutter.dart';

import '../widgets/common.dart';
import '../zoo.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final _input = TextEditingController();
  final _messages = <ChatMessage>[];
  ZooModel _model = chatModels.first;
  TextGenerator? _llm;
  String? _loaded;
  DownloadProgress? _progress;
  String _reply = '';
  String _status = 'The model downloads the first time you send a message.';
  bool _busy = false;

  @override
  void dispose() {
    _llm?.close();
    _input.dispose();
    super.dispose();
  }

  Future<TextGenerator> _load() async {
    final current = _llm;
    if (current != null && _loaded == _model.id) return current;
    await current?.close();
    _llm = null;
    setState(() => _status = 'Loading ${_model.title}…');
    final llm = await TextGenerator.load(
      _model.location,
      onProgress: (p) => setState(() => _progress = p),
    );
    _llm = llm;
    _loaded = _model.id;
    setState(() => _status = '${_model.title} · ${llm.model.variant.id}');
    return llm;
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    if (text.isEmpty || _busy) return;
    _input.clear();
    setState(() {
      _messages.add(ChatMessage.user(text));
      _reply = '';
      _busy = true;
    });
    try {
      final llm = await _load();
      final watch = Stopwatch()..start();
      var pieces = 0;
      await for (final piece in llm.chat(_messages)) {
        pieces++;
        setState(() => _reply += piece);
      }
      final seconds = watch.elapsedMilliseconds / 1000;
      setState(() {
        _messages.add(ChatMessage.assistant(_reply.trim()));
        _status =
            '${_model.title} · ${(pieces / seconds).toStringAsFixed(1)} pieces/s';
      });
    } on Object catch (error) {
      if (mounted) showError(context, error);
    } finally {
      if (mounted) {
        setState(() {
          _reply = '';
          _busy = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) => Column(
    children: [
      Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
        child: DropdownButton<ZooModel>(
          value: _model,
          isExpanded: true,
          items: [
            for (final m in chatModels)
              DropdownMenuItem(value: m, child: Text('${m.title} · ${m.note}')),
          ],
          onChanged: _busy ? null : (m) => setState(() => _model = m!),
        ),
      ),
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: StatusLine(text: _status, progress: _progress),
      ),
      Expanded(
        child: ListView(
          padding: const EdgeInsets.all(12),
          children: [
            for (final m in _messages)
              _Bubble(text: m.content, mine: m.role == ChatRole.user),
            if (_reply.isNotEmpty) _Bubble(text: _reply, mine: false),
          ],
        ),
      ),
      SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(8),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _input,
                  onSubmitted: (_) => _send(),
                  decoration: const InputDecoration(
                    hintText: 'Ask something',
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              _busy
                  ? IconButton.filledTonal(
                      onPressed: () => _llm?.cancel(),
                      icon: const Icon(Icons.stop),
                    )
                  : IconButton.filled(
                      onPressed: _send,
                      icon: const Icon(Icons.send),
                    ),
            ],
          ),
        ),
      ),
    ],
  );
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.text, required this.mine});

  final String text;
  final bool mine;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Align(
      alignment: mine ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(12),
        constraints: const BoxConstraints(maxWidth: 320),
        decoration: BoxDecoration(
          color: mine
              ? scheme.primaryContainer
              : scheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Text(text),
      ),
    );
  }
}
