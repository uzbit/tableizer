import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import '../detection_box.dart';
import '../services/table_detection_result.dart';
import '../widgets/table_ball_painter.dart';

class BallResultsScreen extends StatelessWidget {
  final List<Detection> ballDetections;
  final ui.Size? capturedImageSize;
  final TableDetectionResult? tableDetectionResult;

  const BallResultsScreen({
    super.key,
    required this.ballDetections,
    this.capturedImageSize,
    this.tableDetectionResult,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Ball Analysis Results (${ballDetections.length} balls)'),
        backgroundColor: Colors.green.shade800,
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Pool table background - scale width to screen width, keep aspect ratio
          Image.asset(
            'assets/images/shotstudio_table_felt_only.png',
            width: double.infinity,
            fit: BoxFit.fitWidth,
          ),
          
          // Ball positions overlay
          if (ballDetections.isNotEmpty && capturedImageSize != null)
            LayoutBuilder(
              builder: (context, constraints) {
                final screenWidth = constraints.maxWidth;
                final screenHeight = constraints.maxHeight;
                
                // Calculate same table dimensions as above (rotated table)
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
                    child: CustomPaint(
                      size: Size(tableWidth, tableHeight),
                      painter: TableBallPainter(
                        detections: ballDetections,
                        capturedImageSize: capturedImageSize,
                        tableDisplaySize: Size(tableWidth, tableHeight),
                        tableDetectionResult: tableDetectionResult,
                      ),
                    ),
                  ),
                );
              },
            ),
          
        ],
      ),
    );
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