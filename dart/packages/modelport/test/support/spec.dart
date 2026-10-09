import 'dart:io';

import 'package:path/path.dart' as p;

/// The repository's `spec/` folder, found by walking up from the current directory.
Directory specDir() {
  var dir = Directory.current.absolute;
  while (true) {
    final candidate = Directory(p.join(dir.path, 'spec'));
    if (File(p.join(candidate.path, 'manifest.schema.json')).existsSync()) {
      return candidate;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError(
        'spec/ folder not found above ${Directory.current.path}',
      );
    }
    dir = parent;
  }
}

String readSpecFile(String relative) =>
    File(p.join(specDir().path, relative)).readAsStringSync();
