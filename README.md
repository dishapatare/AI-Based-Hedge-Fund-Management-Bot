# AI-Based-Hedge-Fund-Management-Bot

# Project Overview
This project presents an end-to-end system for intelligent hedge fund management using advanced AI models. The application integrates:

  - A BiLSTM-based prediction model trained on real-time financial data.
  - A Flask-based backend API for model inference and live stock data delivery.
  - And a Flutter mobile application to visualize market trends, provide trading recommendations, and support automated or manual trading.

It enables users to monitor live stock trends, receive AI-based buy/sell predictions, and automate trades in real-time, forming a seamless pipeline from data acquisition to decision execution.



## Colab Notebook
[Colab Notebook](https://colab.research.google.com/drive/1Dkt-Eo06n26HnpXjIdMhxpJtA10MYs6i?usp=sharing)

## Google Drive (Trained Models)
[Google Drive (Trained Models)](https://drive.google.com/drive/folders/1D_iRw_Ks9xVeRgONqDGDqbemmq17OvR3?usp=sharing)

## Flask API
- **Python Version:** 3.11.0
- **Libraries:**
  - numpy==2.0.2
  - pandas==2.2.2
  - seaborn==0.13.2
  - yfinance==0.2.55
  - scikit-learn==1.6.1
  - tensorflow==2.18.0
  - flask==3.1.0
  - flask_cors==5.0.1
  - SQLAlchemy==2.0.40
  - flask_sqlalchemy==3.1.1

## Flutter Mobile Application
- **Environment:**
  - sdk: ^3.5.4
- **Dependencies:**
  - flutter:
    - sdk: flutter
  - cupertino_icons: ^1.0.8
  - http: ^0.13.0
  - fl_chart: ^0.60.0
  - intl: ^0.17.0

## Development Environment

### Flutter Version
```plaintext
Flutter 3.24.5 • channel stable • https://github.com/flutter/flutter.git
Framework • revision dec2ee5c1f (5 months ago) • 2024-11-13 11:13:06 -0800
Engine • revision a18df97ca5
Tools • Dart 3.5.4 • DevTools 2.37.3
```

### Gradle Version
```plaintext
------------------------------------------------------------
Gradle 8.3
------------------------------------------------------------
Build time:   2023-08-17 07:06:47 UTC
Revision:     8afbf24b469158b714b36e84c6f4d4976c86fcd5
Kotlin:       1.9.0
Groovy:       3.0.17
Ant:          Apache Ant(TM) version 1.10.13 compiled on January 4 2023
JVM:          17.0.11 (Oracle Corporation 17.0.11+7-LTS-207)
OS:           Windows 11 10.0 amd64
```

### Flutter Doctor Output
```plaintext
[√] Flutter (Channel stable, 3.24.5, on Microsoft Windows [Version 10.0.22631.5189], locale en-IN)
    • Flutter version 3.24.5 on channel stable at O:\Installed\flutter_windows_3.22.2-stable\flutter
    • Upstream repository https://github.com/flutter/flutter.git
    • Framework revision dec2ee5c1f (5 months ago), 2024-11-13 11:13:06 -0800
    • Engine revision a18df97ca5
    • Dart version 3.5.4
    • DevTools version 2.37.3

[√] Windows Version (Installed version of Windows is version 10 or higher)

[√] Android toolchain - develop for Android devices (Android SDK version 35.0.0)
    • Android SDK at O:\Installed\AndroidSDK
    • Platform android-35, build-tools 35.0.0
    • Java binary at: O:\Installed Soft\JDK17\bin\java
    • Java version Java(TM) SE Runtime Environment (build 17.0.11+7-LTS-207)
    • All Android licenses accepted.

[√] Chrome - develop for the web
    • Chrome at C:\Program Files\Google\Chrome\Application\chrome.exe

[X] Visual Studio - develop Windows apps
    X Visual Studio not installed; this is necessary to develop Windows apps.
      Download at https://visualstudio.microsoft.com/downloads/.
      Please install the "Desktop development with C++" workload, including all of its default components

[√] Android Studio (version 2024.2)
    • Android Studio at O:\Installed\AndroidStudio
    • Flutter plugin can be installed from:
       https://plugins.jetbrains.com/plugin/9212-flutter
    • Dart plugin can be installed from:
       https://plugins.jetbrains.com/plugin/6351-dart
    • Java version OpenJDK Runtime Environment (build 21.0.3+-12282718-b509.11)

[√] VS Code (version 1.99.2)
    • VS Code at C:\Users\omvas\AppData\Local\Programs\Microsoft VS Code
    • Flutter extension version 3.108.0

[√] Connected device (3 available)
    • Windows (desktop) • windows • windows-x64    • Microsoft Windows [Version 10.0.22631.5189]
    • Chrome (web)      • chrome  • web-javascript • Google Chrome 135.0.7049.85
    • Edge (web)        • edge    • web-javascript • Microsoft Edge 135.0.3179.66

[√] Network resources
    • All expected network resources are available.

! Doctor found issues in 1 category.
```

## Steps to Run the Project
1. Make sure you install everything mentioned above with their specific versions.
2. Run the API (`API/main.py`) file.
3. Copy the Public IP on which the API is running from the terminal.
4. Open `'/TradingBot'` in Android Studio and replace that IP with `api_url` in `TradingBot/lib/globals.dart`.
5. Now run the Flutter project.
>>>>>>> acc74fe (Files Added)
