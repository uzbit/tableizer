import 'package:flutter/material.dart';
import '../services/table_detection_service.dart';

class TableScreen extends StatefulWidget {
  final TableDetectionService tableDetectionService;

  const TableScreen({super.key, required this.tableDetectionService});

  @override
  State<TableScreen> createState() => _TableScreenState();
}

class _TableScreenState extends State<TableScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Table Detection'),
      ),
      body: const Center(
        child: Text('Table detection results will be displayed here.'),
      ),
    );
  }
}
