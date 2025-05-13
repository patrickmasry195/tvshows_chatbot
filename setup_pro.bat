@echo off
if not exist "moviebot_env" (
    echo Creating virtual environment...
    python -m venv moviebot_env
)

call moviebot_env\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing required libraries...
pip install flask flask_cors pandas numpy transformers pyspellchecker rapidfuzz scikit-learn nltk torch torchvision torchaudio

echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt')"

echo Starting the Flask server...
python app.py

deactivate

echo Virtual environment setup and server start complete.
pause
