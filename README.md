# Deep Learning Toolbox

Interactive Flask application for learning and experimenting with deep learning concepts through visual dashboards and trainer APIs.

## What Is Included

- Perceptron trainer
- Backpropagation trainer
- MLP trainer
- Activation functions explorer
- Digit detection (MNIST)
- CNN visual labs
- OpenCV object/person detection
- RNN trainer
- Sentiment analysis (LSTM-based text classification)

## Directory Structure

```text
deep_learning_toolbox/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html
│   ├── perceptron_dashboard.html
│   ├── backpropogation_dashboard.html
│   ├── mlp_dashboard.html
│   ├── activation_functions_dashboard.html
│   ├── digit_detection_dashboard.html
│   ├── cnn_dashboard.html
│   ├── opencv_dashboard.html
│   ├── rnn_dashboard.html
│   └── sentiment_analysis_dashboard.html
├── static/
│   ├── css/
│   │   ├── style.css
│   │   ├── perceptron-dashboard.css
│   │   ├── backprop-dashboard.css
│   │   ├── mlp-dashboard.css
│   │   ├── activation-functions-dashboard.css
│   │   ├── digit-detection-dashboard.css
│   │   ├── cnn-dashboard.css
│   │   ├── opencv-dashboard.css
│   │   ├── rnn-dashboard.css
│   │   └── sentiment-dashboard.css
│   └── js/
│       ├── main.js
│       ├── gates-training.js
│       ├── perceptron-trainer.js
│       ├── backprop-trainer.js
│       ├── mlp-trainer.js
│       ├── activation-functions-trainer.js
│       ├── digit-detection-trainer.js
│       └── sentiment-trainer.js
├── perceptron/
├── backpropogation/
├── mlp/
├── activation_functions/
├── cnn/
├── opencv/
├── rnn/
├── rnn_project/
│   ├── sentiment_engine.py
│   └── testdata.manual.2009.06.14.csv
└── yolov8n.pt
```

## Requirements

- Python 3.10+
- pip
- Windows/macOS/Linux

## Setup And Run

### 0. Clone repository

```bash
git clone https://github.com/jai050803/deep_learning_toolbox.git
cd deep_learning_toolbox
```

### 1. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
python app.py
```

Open:

- http://127.0.0.1:5000

## Main Routes

- `/` dashboard
- `/perceptron`
- `/backpropogation`
- `/mlp`
- `/activation_functions`
- `/digit_detection`
- `/cnn`
- `/opencv_project`
- `/rnn_project`
- `/sentiment_analysis`

## Sentiment Analysis Notes

- Engine file: `rnn_project/sentiment_engine.py`
- Dataset default: `rnn_project/testdata.manual.2009.06.14.csv`
- API train endpoint: `POST /api/sentiment/train`
- API predict endpoint: `POST /api/sentiment/predict`

## API Testing Quick Examples

Train:

```bash
curl -X POST http://127.0.0.1:5000/api/sentiment/train \
    -H "Content-Type: application/json" \
    -d "{\"epochs\":20,\"batch_size\":16,\"vocab_size\":5000,\"embedding_dim\":64,\"lstm_units\":128,\"dense_units\":64,\"max_len\":100,\"test_split\":0.2}"
```

Predict:

```bash
curl -X POST http://127.0.0.1:5000/api/sentiment/predict \
    -H "Content-Type: application/json" \
    -d "{\"model_id\":\"sentiment_model_0\",\"text\":\"this is a great movie\"}"
```

## Troubleshooting

- If import fails, confirm venv is active and dependencies are installed.
- If TensorFlow warnings appear about oneDNN, they are informational.
- If sentiment prediction says model not found, train once first from dashboard or API.
- If port 5000 is busy, stop previous process or change `port` in `app.py`.

## Git Ignore Policy

This repository excludes non-essential local/runtime files such as:

- Virtual environments (`.venv/`, `env/`)
- Python cache folders (`__pycache__/`)
- Notebook checkpoints (`.ipynb_checkpoints/`)
- Temporary logs and build outputs
- Very large optional raw dataset files not needed for app execution

## Production Notes

- Set `DEBUG = False` in `app.py`
- Move `SECRET_KEY` to environment variable
- Run behind a production WSGI server (for example, gunicorn or waitress)
