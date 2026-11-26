# DeepCal NLP

An AI-powered voice-based food calorie estimation and nutrition assistant that helps users track their food intake through natural language processing and voice interaction.

## Overview

DeepCal NLP is a voice-enabled nutrition assistant designed to simplify food tracking. It uses Natural Language Processing (NLP) to understand food-related queries, extract food items and quantities, and match them with nutritional information.

The application allows users to interact with a food assistant using natural language, making calorie tracking more convenient and accessible.

## Features

- Voice interaction for food-related queries.
- Natural Language Processing to understand food-related requests.
- Food matching and quantity extraction.
- Intent recognition using an NLP-based classification model.
- Nutrition-related queries and calorie estimation.
- REST API backend built using FastAPI.
- Web interface for interacting with the assistant.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| FastAPI | Backend API |
| NLP | Food query understanding |
| Scikit-learn / Joblib | Intent classification and model handling |
| HTML, CSS, JavaScript | Web interface |
| CSV | Food and training datasets |
| Docker | Containerization |

## Project Structure

```text
deepcal/
├── app/
│   ├── core.py
│   ├── main.py
│   ├── tools.py
│   ├── voice.py
│   └── nlp/
├── data/
├── models/
├── static/
├── tests/
├── .env.example
├── .gitignore
├── Dockerfile
├── Procfile
└── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Gaurikul07/DeepCal-NLP.git
cd DeepCal-NLP/deepcal
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file from the example configuration:

```powershell
Copy-Item .env.example .env
```

Add your own API credentials to `.env` if required.

**Never upload your actual `.env` file or API keys to GitHub.**

### 5. Run the application

```bash
uvicorn app.main:app --reload
```

Open your browser and visit:

http://127.0.0.1:8000

If FastAPI documentation is enabled, access it at:

http://127.0.0.1:8000/docs

## Running Tests

Run the test suite using:

```bash
pytest
```

## Environment Variables

Configure the environment variables required by the application in your `.env` file.

Refer to `.env.example` for the expected variable names.

Do not commit real API keys, passwords, or other sensitive credentials.

## Future Improvements

- Improve food recognition and calorie estimation accuracy.
- Expand the food and nutrition database.
- Add personalized nutrition recommendations.
- Improve voice recognition and speech response.
- Develop a user-friendly nutrition tracking dashboard.
