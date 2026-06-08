       # Flight Turbulence Prediction System

A comprehensive system for predicting and analyzing turbulence at Indian airports using machine learning and real-time weather data.

## Quick Start

1. Make sure you have Python 3.11 or higher installed:
in project directory, run
python --version

2. Install dependencies:
in project directory, to install dependencies, run: 
pip install -r requirements.txt


3. To activate virtual environment, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

4. To run the application now:
streamlit run app.py

The app will open automatically in your default web browser at `http://localhost:8501`

## Features

- Airport Map
- Weather Analysis
- ML Prediction
- Route Assessment
- Historical Analysis
- Real-Time Prediction
- Atmospheric Visualization
- Data Upload
- Statistical Dashboard
- Alert System
- Severity Classification
- Deep Learning Prediction

## Environment Variables

The following environment variables are automatically loaded from `.env` file:

- `OPENWEATHERMAP_API_KEY`: For weather data (already configured)
- `DATABASE_URL`: Optional, for persistent storage (defaults to demo mode if not set)

## Demo Mode

If no database URL is provided, the system runs in demo mode with in-memory storage, perfect for testing and demonstration purposes.
