// lib/screens/home/home_v1.dart
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../dashboard/dashboard_v1.dart';
import 'package:trading_bot/globals.dart' as globals;

class HomePageV1 extends StatefulWidget {
  const HomePageV1({Key? key}) : super(key: key);

  @override
  _HomePageV1State createState() => _HomePageV1State();
}

class _HomePageV1State extends State<HomePageV1> {
  bool _isLoading = true;
  List<String> _symbols = [];
  String? _selectedSymbol;
  String errorMessage = '';
  bool _autoTrading = false; // auto trading switch state

  @override
  void initState() {
    super.initState();
    fetchSymbols();
  }

  Future<void> fetchSymbols() async {
    setState(() {
      _isLoading = true;
      errorMessage = '';
    });
    try {
      final url = Uri.parse('${globals.api_url}/list_symbols');
      final response = await http.get(url);
      if (response.statusCode == 200) {
        List<dynamic> data = jsonDecode(response.body);
        setState(() {
          _symbols = data.map((item) => item.toString()).toList();
        });
      } else {
        setState(() {
          errorMessage = 'Failed to load symbols.';
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Connection error, please try again.';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _confirmSelection() {
    if (_selectedSymbol == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select a symbol.')),
      );
    } else {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => DashboardPageV1(
            selectedSymbol: _selectedSymbol!,
            autoTrading: _autoTrading,
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Select Symbol')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: _isLoading
              ? const CircularProgressIndicator()
              : errorMessage.isNotEmpty
              ? Text(errorMessage, style: const TextStyle(color: Colors.red))
              : Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Choose a symbol:',
                style: TextStyle(fontSize: 18),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16.0),
              Expanded(
                child: ListView.builder(
                  itemCount: _symbols.length,
                  itemBuilder: (context, index) {
                    final symbol = _symbols[index];
                    return Container(
                      margin: const EdgeInsets.symmetric(vertical: 4.0),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey, width: 0.5),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: RadioListTile<String>(
                        title: Row(
                          children: [
                            const Icon(Icons.show_chart),
                            const SizedBox(width: 10),
                            Text(symbol),
                          ],
                        ),
                        value: symbol,
                        groupValue: _selectedSymbol,
                        onChanged: (value) {
                          setState(() {
                            _selectedSymbol = value;
                          });
                        },
                      ),
                    );
                  },
                ),
              ),
              // Auto Trading Switch list tile
              Container(
                margin: const EdgeInsets.symmetric(vertical: 8.0),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey, width: 0.5),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: ListTile(
                  leading: Switch(
                    value: _autoTrading,
                    onChanged: (value) {
                      setState(() {
                        _autoTrading = value;
                      });
                    },
                  ),
                  title: const Text('Auto Trading'),
                ),
              ),
              const SizedBox(height: 16.0),
              ElevatedButton(
                onPressed: _confirmSelection,
                child: const Text('Confirm'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
