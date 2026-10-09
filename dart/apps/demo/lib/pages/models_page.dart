import 'dart:io';

import 'package:flutter/material.dart';
import 'package:modelport_flutter/modelport_flutter.dart';
import 'package:path/path.dart' as p;

class ModelsPage extends StatefulWidget {
  const ModelsPage({super.key});

  @override
  State<ModelsPage> createState() => _ModelsPageState();
}

class _ModelsPageState extends State<ModelsPage> {
  Map<String, int> _cached = const {};
  int _total = 0;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    final root = Directory(p.join(ModelPort.store.root.path, 'models'));
    final sizes = <String, int>{};
    if (root.existsSync()) {
      for (final dir in root.listSync().whereType<Directory>()) {
        var size = 0;
        for (final f in dir.listSync(recursive: true).whereType<File>()) {
          size += f.lengthSync();
        }
        sizes[p.basename(dir.path)] = size;
      }
    }
    final total = await ModelPort.store.cacheSize();
    if (mounted) {
      setState(() {
        _cached = sizes;
        _total = total;
      });
    }
  }

  String _mb(int bytes) => '${(bytes / 1e6).toStringAsFixed(1)} MB';

  @override
  Widget build(BuildContext context) => RefreshIndicator(
    onRefresh: _refresh,
    child: ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: ListTile(
            leading: const Icon(Icons.memory),
            title: Text('Device RAM: ${ModelPort.deviceRamMb ?? 'unknown'} MB'),
            subtitle: Text(
              'Engines: ${ModelPort.adapters.map((a) => a.runtime).join(', ')}',
            ),
          ),
        ),
        ListTile(
          title: const Text('Downloaded models'),
          trailing: Text(_mb(_total)),
        ),
        if (_cached.isEmpty)
          const ListTile(title: Text('Nothing downloaded yet.')),
        for (final MapEntry(key: id, value: size) in _cached.entries)
          ListTile(
            leading: const Icon(Icons.inventory_2_outlined),
            title: Text(id),
            subtitle: Text(_mb(size)),
            trailing: IconButton(
              icon: const Icon(Icons.delete_outline),
              tooltip: 'Delete',
              onPressed: () async {
                await ModelPort.store.delete(id);
                await _refresh();
              },
            ),
          ),
      ],
    ),
  );
}
