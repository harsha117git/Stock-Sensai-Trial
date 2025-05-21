# Creating an Executable from the Stock Price Prediction App

This guide explains how to package the Stock Price Prediction Streamlit application into a standalone executable that can be distributed to users who don't have Python installed.

## Prerequisites

- A complete working version of the application
- Windows, macOS, or Linux operating system
- Python installed on the packaging machine (not needed for end users)

## Quick Start

1. Make sure your application is working correctly
2. Run the packaging script:
   ```
   python package_app.py
   ```
3. Wait for the packaging process to complete (may take several minutes)
4. Distribute the generated ZIP file to users

## For Users Running the Executable

Users who receive the packaged application should:

1. Extract the ZIP file to a directory of their choice
2. Run the application:
   - On Windows: Double-click `start_app.bat`
   - On macOS/Linux: Run `./start_app.sh` in terminal
3. A browser will automatically open with the application running

## Manual Packaging Steps

If the automatic packaging script fails, you can manually package the application:

### 1. Install PyInstaller

```
pip install pyinstaller
```

### 2. Create a .streamlit Directory

Create a directory called `.streamlit` and add a `config.toml` file with:

```
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
```

### 3. Create a PyInstaller Spec File

Create `stock_app.spec` with the content from the packaging script.

### 4. Run PyInstaller

```
pyinstaller stock_app.spec --clean
```

### 5. Create Startup Scripts

Create `start_app.bat` (Windows) or `start_app.sh` (macOS/Linux) that will launch the application and open a browser.

### 6. Package for Distribution

Collect all the necessary files into a ZIP archive for distribution.

## Troubleshooting

Common issues during packaging:

1. **Missing dependencies**: Add them to the `hiddenimports` section in the spec file
2. **File not found errors**: Make sure all resource files are included in the `datas` section
3. **Startup issues**: Check that the path to the executable is correct in the startup scripts

## Notes for Developers

- The executable will be significantly larger than the source code
- The package includes all Python dependencies
- The startup script sets necessary environment variables and opens a browser
- Database files are included in the package; users' data will be stored locally