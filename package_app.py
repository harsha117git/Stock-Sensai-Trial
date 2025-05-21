import os
import subprocess
import sys
import shutil
import zipfile
import platform

print("Stock Price Prediction App Packager")
print("===================================")
print("This script will package the app into an executable file")

# Determine the operating system
operating_system = platform.system().lower()
print(f"Detected operating system: {operating_system}")

# Install required packaging tools
print("\nInstalling packaging tools...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

# Create spec file for PyInstaller
print("\nCreating spec file for PyInstaller...")

spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['auth_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('data', 'data'),
        ('.streamlit', '.streamlit'),
    ],
    hiddenimports=[
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'scikit-learn',
        'yfinance',
        'tensorflow',
        'requests',
        'bs4',
        'nltk',
        'matplotlib',
        'statsmodels',
        'sqlalchemy',
        'streamlit_authenticator',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StockPricePrediction',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StockPricePrediction',
)
"""

with open('stock_app.spec', 'w') as f:
    f.write(spec_content)

# Create startup script
print("\nCreating startup script...")
if operating_system == 'windows':
    startup_content = """@echo off
echo Starting Stock Price Prediction Application...
cd %~dp0
set STREAMLIT_SERVER_PORT=8501
set STREAMLIT_SERVER_HEADLESS=true
start http://localhost:8501
StockPricePrediction\\StockPricePrediction.exe
"""
    with open('start_app.bat', 'w') as f:
        f.write(startup_content)
else:  # Linux/Mac
    startup_content = """#!/bin/bash
echo "Starting Stock Price Prediction Application..."
cd "$(dirname "$0")"
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
xdg-open http://localhost:8501 2>/dev/null || open http://localhost:8501 2>/dev/null &
./StockPricePrediction/StockPricePrediction
"""
    with open('start_app.sh', 'w') as f:
        f.write(startup_content)
    # Make it executable
    os.chmod('start_app.sh', 0o755)

# Create .streamlit directory if it doesn't exist
if not os.path.exists('.streamlit'):
    os.makedirs('.streamlit')

# Create Streamlit config
print("\nCreating Streamlit configuration...")
streamlit_config = """
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
"""

with open('.streamlit/config.toml', 'w') as f:
    f.write(streamlit_config)

# Build the executable
print("\nBuilding the executable (this may take a while)...")
try:
    subprocess.check_call(['pyinstaller', 'stock_app.spec', '--clean'])
    print("\nExecutable successfully built!")
    
    # Package the distribution
    print("\nCreating distributable package...")
    
    package_name = f"StockPricePrediction_{operating_system}"
    
    # Create a distributable directory
    if os.path.exists(package_name):
        shutil.rmtree(package_name)
    os.makedirs(package_name)
    
    # Copy files to the distribution directory
    shutil.copytree('dist/StockPricePrediction', f"{package_name}/StockPricePrediction")
    
    if operating_system == 'windows':
        shutil.copy('start_app.bat', package_name)
    else:
        shutil.copy('start_app.sh', package_name)
    
    # Create user instructions
    instructions = f"""
Stock Price Prediction Application
=================================

Installation Instructions:
1. Extract this package to a directory of your choice
2. Run the application by using:
   - On Windows: Double-click on 'start_app.bat'
   - On Mac/Linux: Open terminal in this directory and type './start_app.sh'
3. A browser window should open automatically. If not, open your browser and go to: http://localhost:8501

Requirements:
- No Python installation required (all dependencies are bundled)
- Internet connection for fetching stock data

For support, please contact the application provider.
"""
    
    with open(f"{package_name}/README.txt", 'w') as f:
        f.write(instructions)
    
    # Create ZIP package
    print(f"\nCreating ZIP archive of the package...")
    shutil.make_archive(package_name, 'zip', '.', package_name)
    
    print(f"\nPackage created successfully: {package_name}.zip")
    print(f"You can distribute this ZIP file to users.")
    
except Exception as e:
    print(f"\nError during packaging: {str(e)}")
    print("Please check if PyInstaller is properly installed and try again.")

print("\nPackaging process completed.")