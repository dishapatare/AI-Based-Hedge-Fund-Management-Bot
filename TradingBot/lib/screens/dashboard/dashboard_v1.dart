import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'package:trading_bot/globals.dart' as globals;

class DashboardPageV1 extends StatefulWidget {
  final String selectedSymbol;
  final bool autoTrading; // if true, trades execute automatically

  const DashboardPageV1({
    Key? key,
    required this.selectedSymbol,
    required this.autoTrading,
  }) : super(key: key);

  @override
  _DashboardPageV1State createState() => _DashboardPageV1State();
}

class _DashboardPageV1State extends State<DashboardPageV1> {
  // Chart data variables
  List<FlSpot> _chartSpots = [];
  double _minX = 0, _maxX = 1, _minY = 0, _maxY = 1;
  String _chartError = '';
  bool _chartLoading = false;
  bool _isChartFetching = false;
  Timer? _chartTimer;
  bool _firstChartFetched = false;

  // Prediction variables
  String _predictedDirection = '';
  String _predictedPrice = '';
  String _predictionError = '';
  bool _predictionLoading = false;
  bool _isPredictionFetching = false;

  // New: Last price from prediction
  double _lastPrice = 0.0;

  // Trading simulation variables
  double _currentBalance = 10000.0; // starting balance
  String _position = "none"; // "none", "long", or "short"

  // Timer for chart updates (every 30 sec) and predictions (every 15 min)
  Timer? _predictTimer;

  // Formatter for x-axis time stamps.
  final DateFormat _timeFormatter = DateFormat('HH:mm');

  @override
  void initState() {
    super.initState();
    // Immediately fetch chart data.
    _fetchChartData();

    // Set chart update timer every 30 seconds.
    _chartTimer = Timer.periodic(const Duration(seconds: 27), (timer) {
      _fetchChartData();
    });

    // Set prediction timer every 15 minutes.
    _predictTimer = Timer.periodic(const Duration(minutes: 2), (timer) {
      if (_firstChartFetched) {
        _fetchPrediction();
      }
    });
  }

  @override
  void dispose() {
    _chartTimer?.cancel();
    _predictTimer?.cancel();
    super.dispose();
  }

  // Lock-based fetch for chart data.
  Future<void> _fetchChartData() async {
    while (_isChartFetching) {
      await Future.delayed(const Duration(milliseconds: 100));
    }
    _isChartFetching = true;
    setState(() {
      _chartLoading = true;
      _chartError = '';
    });
    try {
      final uri = Uri.parse(
          '${globals.api_url}/get_live_data?ticker=${widget.selectedSymbol}&token=${globals.token}');
      final response = await http.get(uri);
      if (response.statusCode == 200) {
        List<dynamic> data = jsonDecode(response.body);
        List<FlSpot> spots = [];
        for (var item in data) {
          DateTime dt = DateTime.parse(item['Datetime']);
          double x = dt.millisecondsSinceEpoch.toDouble();
          double y = (item['Close'] is int)
              ? (item['Close'] as int).toDouble()
              : (item['Close'] as double);
          spots.add(FlSpot(x, y));
        }
        spots.sort((a, b) => a.x.compareTo(b.x));
        if (spots.isNotEmpty) {
          setState(() {
            _chartSpots = spots;
            _minX = spots.first.x;
            _maxX = spots.last.x;
            _minY = spots.map((e) => e.y).reduce((a, b) => a < b ? a : b);
            _maxY = spots.map((e) => e.y).reduce((a, b) => a > b ? a : b);
          });
        }
      } else {
        setState(() {
          _chartError = 'Failed to load live data.';
        });
      }
    } catch (e) {
      setState(() {
        _chartError = 'Connection error while fetching live data.';
      });
    } finally {
      setState(() {
        _chartLoading = false;
      });
      _isChartFetching = false;

      // On the first successful chart fetch, trigger a prediction request.
      if (!_firstChartFetched && _chartError.isEmpty) {
        _firstChartFetched = true;
        _fetchPrediction();
      }
    }
  }

  // Lock-based fetch for prediction data.
  Future<void> _fetchPrediction() async {
    while (_isPredictionFetching) {
      await Future.delayed(const Duration(milliseconds: 100));
    }
    _isPredictionFetching = true;
    setState(() {
      _predictionLoading = true;
      _predictionError = '';
    });
    try {
      final uri = Uri.parse(
          '${globals.api_url}/predict?ticker=${widget.selectedSymbol}&token=${globals.token}');
      final response = await http.get(uri);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _predictedDirection = data['predicted_direction'] ?? '';
          _predictedPrice = data['predicted_price'] ?? '';
          // Save last price from prediction (convert to double).
          _lastPrice = double.tryParse(data['last_price'] ?? "0") ?? 0.0;
        });
      } else {
        final data = jsonDecode(response.body);
        setState(() {
          _predictionError = data['error'] ?? 'Prediction failed.';
        });
      }
    } catch (e) {
      setState(() {
        _predictionError = 'Connection error while fetching prediction.';
      });
    } finally {
      setState(() {
        _predictionLoading = false;
      });
      _isPredictionFetching = false;

      // If auto trading is enabled, execute trade automatically.
      if (widget.autoTrading) {
        _executeTrade();
      }
    }
  }

  // Trade execution based on prediction.
  // This function first closes any open trade and then opens a new trade based on the current prediction.
  void _executeTrade() {
    if (_lastPrice <= 0) return; // invalid price, skip.

    // Close any open position.
    if (_position == "long") {
      // Close long: sell at current last price.
      _currentBalance += _lastPrice;
    } else if (_position == "short") {
      // Close short: buy at current last price.
      _currentBalance -= _lastPrice;
    }

    // Open new position based on predicted direction.
    if (_predictedDirection.toUpperCase() == "UP") {
      // Buy: subtract last price from wallet.
      _currentBalance -= _lastPrice;
      _position = "long";
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Transaction: (UP) Position 'Long'")),);
    } else if (_predictedDirection.toUpperCase() == "DOWN") {
      // Sell: add last price to wallet.
      _currentBalance += _lastPrice;
      _position = "short";
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Transaction: (DOWN) Position 'Short'")),);
    }

    // Update UI with new balance.
    setState(() {
      // _currentBalance changed and profit/loss can be computed as _currentBalance - 10000.
    });
  }

  // When auto trading is disabled, user can press this button to manually execute a trade.
  void _manualTrade() {
    _executeTrade();
  }

  // Build the line chart widget.
  Widget _buildLineChart() {
    if (_chartSpots.isEmpty) {
      return const Center(child: Text('No chart data available.'));
    }
    return LineChart(
      LineChartData(
        lineTouchData: LineTouchData(
          handleBuiltInTouches: true,
          touchTooltipData: LineTouchTooltipData(
            tooltipBgColor: Colors.blueGrey.withOpacity(0.8),
            getTooltipItems: (List<LineBarSpot> touchedSpots) {
              return touchedSpots.map((spot) {
                DateTime dt = DateTime.fromMillisecondsSinceEpoch(spot.x.toInt());
                return LineTooltipItem(
                  'Time: ${_timeFormatter.format(dt)}\nClose: ${spot.y.toStringAsFixed(2)}',
                  const TextStyle(color: Colors.white),
                );
              }).toList();
            },
          ),
        ),
        minX: _minX,
        maxX: _maxX,
        minY: _minY,
        maxY: _maxY,
        gridData: FlGridData(show: true),
        borderData: FlBorderData(
          show: true,
          border: const Border(
            left: BorderSide(color: Colors.black),
            bottom: BorderSide(color: Colors.black),
            top: BorderSide(color: Colors.grey),
            right: BorderSide(color: Colors.grey),
          ),
        ),
        titlesData: FlTitlesData(
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 40,
              interval: ((_maxX - _minX) / 4),
              getTitlesWidget: (value, meta) {
                DateTime dt = DateTime.fromMillisecondsSinceEpoch(value.toInt());
                return Padding(
                  padding: const EdgeInsets.only(top: 8.0, right: 10),
                  child: Text(_timeFormatter.format(dt), style: const TextStyle(fontSize: 10)),
                );
              },
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 40,
              interval: ((_maxY - _minY) / 4),
              getTitlesWidget: (value, meta) {
                return Text(value.toStringAsFixed(2), style: const TextStyle(fontSize: 10));
              },
            ),
          ),
          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        lineBarsData: [
          LineChartBarData(
            spots: _chartSpots,
            isCurved: true,
            barWidth: 2,
            color: Colors.blue,
            dotData: FlDotData(show: false),
          ),
        ],
      ),
      swapAnimationDuration: const Duration(milliseconds: 250),
    );
  }

  // Build the prediction view as a bordered ListTile.
  Widget _buildPredictionTile() {
    // Choose leading icon and button label based on predicted_direction.
    Icon leadingIcon;
    String tradeLabel;
    if (_predictedDirection.toUpperCase() == "UP") {
      leadingIcon = const Icon(Icons.arrow_upward, color: Colors.green);
      tradeLabel = "Buy";
    } else if (_predictedDirection.toUpperCase() == "DOWN") {
      leadingIcon = const Icon(Icons.arrow_downward, color: Colors.red);
      tradeLabel = "Sell";
    } else {
      leadingIcon = const Icon(Icons.help_outline);
      tradeLabel = "";
    }

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      decoration: BoxDecoration(
          border: Border.all(color: Colors.grey, width: 0.5),
          borderRadius: BorderRadius.circular(4)),
      child: ListTile(
        contentPadding: EdgeInsets.symmetric(horizontal: 8),
        leading: leadingIcon,
        title: Text("Predicted Direction: $_predictedDirection", style: TextStyle(fontWeight: FontWeight.bold),),
        subtitle: Text("Last Price: ${_lastPrice.toStringAsFixed(2)} \nExpected to touch: $_predictedPrice"),
        trailing: widget.autoTrading
            ? const Text("Auto Pilot")
            : ElevatedButton(
          onPressed: _manualTrade,
          child: Text(tradeLabel),
        ),
      ),
    );
  }

  // Build the balance container with two columns: current balance and profit/loss.
  Widget _buildBalanceContainer() {
    double profitLoss = _currentBalance - 10000.0;
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
          color: Colors.grey.shade200,
          border: Border.all(color: Colors.grey, width: 0.5),
          borderRadius: BorderRadius.circular(4)),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text("Current Balance", style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 4.0),
              Text("\$${_currentBalance.toStringAsFixed(2)}"),
            ],
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text("Profit/Loss", style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 4.0),
              Text("\$${profitLoss.toStringAsFixed(2)}",
                  style: TextStyle(color: profitLoss >= 0 ? Colors.green : Colors.red)),
            ],
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Dashboard"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            Text(widget.selectedSymbol, style: TextStyle(fontWeight: FontWeight.bold),),
            SizedBox(height: 10,),
            // Live Chart Section
            Expanded(
              flex: 3,
              child: _chartLoading
                  ? const Center(child: CircularProgressIndicator())
                  : _chartError.isNotEmpty
                  ? Center(child: Text(_chartError, style: const TextStyle(color: Colors.red)))
                  : _buildLineChart(),
            ),
            const SizedBox(height: 16.0),
            // Prediction Section (ListTile view)
            _predictionLoading
                ? const CircularProgressIndicator()
                : _buildPredictionTile(),
            const SizedBox(height: 16.0),
            // Balance Section
            _buildBalanceContainer(),
          ],
        ),
      ),
    );
  }
}
