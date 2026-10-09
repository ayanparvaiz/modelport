import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:modelport_flutter/modelport_flutter.dart';

/// The bundled sample photo, a Samoyed.
Future<Uint8List> samplePhoto() async =>
    Uint8List.sublistView(await rootBundle.load('assets/images/samoyed.jpg'));

/// Lets the user pick a photo from the gallery. Null if they cancel.
Future<Uint8List?> pickPhoto() async {
  final file = await ImagePicker().pickImage(source: ImageSource.gallery);
  return file?.readAsBytes();
}

/// A line of status text with an optional download bar.
class StatusLine extends StatelessWidget {
  const StatusLine({super.key, required this.text, this.progress});

  final String text;
  final DownloadProgress? progress;

  @override
  Widget build(BuildContext context) {
    final p = progress;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (p != null && p.fraction < 1) ...[
          LinearProgressIndicator(value: p.fraction),
          const SizedBox(height: 4),
          Text(
            'Downloading ${(p.fraction * 100).toStringAsFixed(0)}% '
            'of ${(p.totalBytes / 1e6).toStringAsFixed(0)} MB',
          ),
        ],
        Text(text, style: Theme.of(context).textTheme.bodySmall),
      ],
    );
  }
}

/// The result of a golden check, green when it passes.
class GoldenCard extends StatelessWidget {
  const GoldenCard({super.key, required this.report});

  final GoldenReport report;

  @override
  Widget build(BuildContext context) => Card(
    color: report.passed ? Colors.green.shade50 : Colors.red.shade50,
    child: ListTile(
      leading: Icon(
        report.passed ? Icons.verified : Icons.error_outline,
        color: report.passed ? Colors.green : Colors.red,
      ),
      title: Text(report.passed ? 'Matches Python' : 'Differs from Python'),
      subtitle: Text(report.toString()),
    ),
  );
}

/// Shows a ModelPortException as a snack bar and its hint, if any.
void showError(BuildContext context, Object error) {
  final text = error is ModelPortException
      ? [error.message, if (error.hint != null) error.hint!].join('\n')
      : '$error';
  ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));
}
