import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:camerawesome/camerawesome_plugin.dart';
import '../models/ball_detection_result.dart';
import '../models/table_detection_result.dart';
import '../widgets/table_ball_painter.dart';
import '../services/table_detection_service.dart';
import 'settings_screen.dart';

class TableResultsScreen extends StatelessWidget {
  final List<BallDetectionResult> ballDetections;
  final ui.Size? capturedImageSize;
  final InputAnalysisImageRotation? capturedRotation;
  final TableDetectionResult? tableDetectionResult;
  final TableDetectionService? tableDetectionService;

  const TableResultsScreen({
    super.key,
    required this.ballDetections,
    this.capturedImageSize,
    this.capturedRotation,
    this.tableDetectionResult,
    this.tableDetectionService,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_buildTitle()),
        backgroundColor: Colors.green.shade800,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (context) => const SettingsScreen(),
                ),
              );
            },
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final screenWidth = constraints.maxWidth;
          final screenHeight = constraints.maxHeight;

          // Calculate table dimensions (1:2 aspect ratio - width:height)
          double tableWidth, tableHeight;

          if (screenWidth * 2 <= screenHeight) {
            // Width constraint: use full width, height = width * 2
            tableWidth = screenWidth;
            tableHeight = screenWidth * 2;
          } else {
            // Height constraint: use full height, width = height / 2
            tableHeight = screenHeight;
            tableWidth = screenHeight / 2;
          }

          return Center(
            child: SizedBox(
              width: tableWidth,
              height: tableHeight,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // Pool table background - now constrained to the same size as overlay
                  Image.asset(
                    'assets/images/shotstudio_table_felt_only.png',
                    fit: BoxFit.fill,
                  ),

                  // Ball positions overlay
                  if (ballDetections.isNotEmpty && capturedImageSize != null)
                    CustomPaint(
                      size: Size(tableWidth, tableHeight),
                      painter: TableBallPainter(
                        detections: ballDetections,
                        capturedImageSize: capturedImageSize,
                        tableDisplaySize: Size(tableWidth, tableHeight),
                        tableDetectionResult: tableDetectionResult,
                        capturedRotation: capturedRotation,
                        transformPointsCallback: tableDetectionService?.transformPoints,
                      ),
                    ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  String _buildTitle() {
    int stripeCount = ballDetections.where((d) => d.classId == 3).length;
    int solidCount = ballDetections.where((d) => d.classId == 2).length;
    
    return 'Tableized - $stripeCount stripes, $solidCount solids';
  }

  String _getClassLabel(int classId) {
    switch (classId) {
      case 0:
        return 'Black';
      case 1:
        return 'Cue';
      case 2:
        return 'Solid';
      case 3:
        return 'Stripe';
      default:
        return 'Ball';
    }
  }
}