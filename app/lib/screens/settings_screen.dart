import 'package:flutter/material.dart';
import '../services/settings_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final SettingsService _settingsService = SettingsService();
  late String _selectedTableSize;

  @override
  void initState() {
    super.initState();
    _selectedTableSize = _settingsService.tableSize;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        backgroundColor: Colors.green.shade800,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Settings',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 30),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Table Size',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                DropdownButton<String>(
                  value: _selectedTableSize,
                  items: const [
                    DropdownMenuItem(
                      value: '9-foot',
                      child: Text('9-foot'),
                    ),
                    DropdownMenuItem(
                      value: '8-foot',
                      child: Text('8-foot'),
                    ),
                    DropdownMenuItem(
                      value: '7-foot',
                      child: Text('7-foot'),
                    ),
                  ],
                  onChanged: (String? newValue) {
                    if (newValue != null) {
                      setState(() {
                        _selectedTableSize = newValue;
                      });
                      _settingsService.setTableSize(newValue);
                    }
                  },
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}