import 'package:flutter/material.dart';

class AppTheme {
  static const Color primaryColor = Color(0xFF1F3C88);
  static const Color secondaryColor = Color(0xFF26A69A);
  static const Color backgroundColor = Color(0xFFF5F5F5);
  static const Color textColor = Color(0xFF212121);
  static const Color whiteColor = Color(0xFFFFFFFF);

  static ThemeData get lightTheme {
    return ThemeData(
      colorScheme: ColorScheme.light(
        primary: primaryColor,
        secondary: secondaryColor,
        background: backgroundColor,
        onPrimary: whiteColor,
        onBackground: textColor,
      ),
      scaffoldBackgroundColor: backgroundColor,
      cardColor: whiteColor,
      appBarTheme: const AppBarTheme(
        backgroundColor: primaryColor,
        elevation: 2,
        titleTextStyle: TextStyle(
          color: whiteColor,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        iconTheme: IconThemeData(color: whiteColor),
      ),
      textTheme: const TextTheme(
        titleLarge: TextStyle(
          fontWeight: FontWeight.bold,
          color: textColor,
          fontSize: 20,
        ),
        bodyMedium: TextStyle(
          color: textColor,
          fontSize: 16,
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: whiteColor,
          textStyle: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
    );
  }
}
