import 'package:flutter/material.dart';
import 'package:trading_bot/screens/login/login_v1.dart';

import 'config/app_theme.dart';

void main() {
  runApp(const TradingBotApp());
}

class TradingBotApp extends StatelessWidget {
  const TradingBotApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Trading Bot',
      theme: AppTheme.lightTheme,
      home: const LoginPageV1(),
    );
  }
}