@echo off
REM Create a virtual environment named "moviebot_env" if it doesn't exist
if not exist "moviebot_env" (
    echo Creating virtual environment...
    python -m venv moviebot_env
)

REM Activate the virtual environment
call moviebot_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install required libraries
echo Installing required libraries...
pip install flask flask_cors pandas numpy transformers pyspellchecker rapidfuzz scikit-learn nltk torch torchvision torchaudio

REM Download required NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt')"

REM Run the Flask server
echo Starting the Flask server...
set FLASK_APP=app.py
set FLASK_ENV=development
flask run

REM Deactivate the virtual environment
deactivate

echo Virtual environment setup and server start complete.
pause
