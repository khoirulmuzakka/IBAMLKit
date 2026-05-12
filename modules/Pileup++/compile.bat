@echo off
setlocal

:: Set project root, build, and output directories
set PROJECT_ROOT=%~dp0
set BUILD_DIR=%PROJECT_ROOT%build
set OUTPUT_DIR=%PROJECT_ROOT%bin

:: Create build and output directories if they don't exist
if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

:: Change to build directory
cd /d "%BUILD_DIR%"

:: Configure with CMake and set output directory
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=%OUTPUT_DIR%\Release
if %errorlevel% neq 0 (
    echo CMake configuration failed.
    exit /b %errorlevel%
)

:: Build the project
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo Build failed.
    exit /b %errorlevel%
)

cmake --install .

endlocal
