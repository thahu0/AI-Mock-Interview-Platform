# AI-Mock-Interview-Platform
The AI Mock Interview Platform is designed to assist institutions in conducting mock interviews with both video and audio support.

It leverages advanced AI models for real-time emotion detection and provides analytical feedback to help candidates improve their performance.
Features

    Video and Audio Interview Modes: Conduct interviews with real-time video/audio streaming.
    Real-Time Emotion Detection: Implemented using DeepFace and Time Distributed CNN-LSTM.
    Speech Recognition (SR): Speech-to-text transcription and analysis using Google APIs.
    Personalized Feedback: Provides confidence metrics, accuracy trends, and improvement tips.

Technologies Used

    Frontend: HTML, CSS, JavaScript (optional: Bootstrap for styling).
    Backend: Python (Flask framework).
    AI Models:
        Time Distributed CNN-LSTM for processing sequential data.
        DeepFace for emotion detection.
    APIs: Google Cloud APIs for speech recognition.
    Database: SQLite or MySQL for storing results (optional).

Installation
Prerequisites

Ensure you have the following installed:

    Python 3.7 or higher
    Flask
    Google Cloud SDK (for Speech Recognition API)
    Required Python libraries (see requirements.txt)

Steps

    Clone the repository:

git clone https://github.com/your-github-repo-link

Navigate to the project directory:

cd AI-Mock-Interview-Platform

Install dependencies:

pip install -r requirements.txt

Set up Google API keys:

    Create a project in the Google Cloud Console.
    Enable the Speech-to-Text API.
    Download the service account key file and save it in the project folder.
    Set the environment variable:

    export GOOGLE_APPLICATION_CREDENTIALS="path-to-your-key-file.json"

Run the application:

python app.py

Access the platform in your browser at:

    http://127.0.0.1:5000

Usage

    Select the mode: Video Interview or Audio Interview.
    Start the session to capture real-time video/audio.
    Answer interview questions displayed on the screen.
    View the results, including:
        Detected emotions (confidence levels).
        Speech-to-text transcription.
        Personalized tips and analysis.

Project Structure

AI-Mock-Interview-Platform/
│
├── app/  
│   ├── static/              # CSS, JavaScript, images  
│   ├── templates/           # HTML templates  
│   ├── routes/              # Flask routes  
│   ├── models.py            # Database models  
│   ├── config.py            # Configuration settings  
│   └── app.py               # Main Flask application  
│--recordings
├── requirements.txt         # Dependencies  
├── README.md                # Project documentation  
└── LICENSE                  # License information  

Contributing

Contributions are welcome! Please fork the repository and submit a pull request. Ensure you follow the coding guidelines provided in the repository.
License

This project is licensed under the MIT License. See the LICENSE file for details.
