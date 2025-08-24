import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import '../detection_box.dart';
import '../widgets/table_ball_painter.dart';

class BallResultsScreen extends StatelessWidget {
  final List<Detection> ballDetections;
  final ui.Size? capturedImageSize;

  const BallResultsScreen({
    super.key,
    required this.ballDetections,
    this.capturedImageSize,
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
                      ),
                    ),
                  ),
                );
              },
            ),
          
          // Bottom panel with ball list
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              height: 150,
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.8),
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(16),
                  topRight: Radius.circular(16),
                ),
              ),
              child: Column(
                children: [
                  Container(
                    padding: const EdgeInsets.all(16),
                    child: Text(
                      'Detected Balls',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  Expanded(
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      itemCount: ballDetections.length,
                      itemBuilder: (context, index) {
                        final detection = ballDetections[index];
                        return Container(
                          margin: const EdgeInsets.only(right: 12),
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Colors.green.withOpacity(0.2),
                            border: Border.all(color: Colors.green),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                _getClassLabel(detection.classId),
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                '${(detection.confidence * 100).toStringAsFixed(1)}%',
                                style: const TextStyle(
                                  color: Colors.white70,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 16),
                ],
              ),
            ),
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