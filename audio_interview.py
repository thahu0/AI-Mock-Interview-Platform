import speech_recognition as sr
import soundfile as sf  
import os
import json
import librosa
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request,Blueprint
from speech_emotion_recognition import speechEmotionRecognition
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from collections import Counter
import json


audio_interview = Blueprint('audio_interview', __name__)

from library.speech_emotion_recognition import *




# Read the overall dataframe before the user starts to add his own data
df = pd.read_csv('static/js/db/histo.txt', sep=",")

# Audio Recording
@audio_interview.route('/audio_interview', methods=("POST", "GET"))
def audio_recording():

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition()

    # Voice Recording
    rec_duration = 60 # in sec
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    
    return render_template('audio_interview.html', display_button=True)




@audio_interview.route('/')
def home():
    return render_template('index.html')

@audio_interview.route('/audio-interview')
def audio_interview_home():
    global questions_and_answers
    questions = load_questions_from_csv()
    audio_selected_questions = questions[:10]

    # Save the selected questions globally or to a temporary file
    with open('audio_selected_questions.json', 'w') as f:
        json.dump(audio_selected_questions, f)

    return render_template('audio_interview.html', questions=[q['question'] for q in audio_selected_questions])

def load_questions_from_csv():
    """
    Load questions from a CSV file.
    """
    import csv
    questions = []
    try:
        with open('questions.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row)
    except FileNotFoundError:
        print("questions.csv file not found.")
    return questions

def load_selected_questions():
    try:
        with open('audio_selected_questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@audio_interview.route('/save_audio', methods=["POST"])
def save_audio():
    try:
        # Get data from request
        question_id = request.form.get("question_id")
        audio_data = request.files['audio']

        # Save the file with a unique name
        save_path = os.path.join('recordings', f'question_{question_id}.wav')
        audio_data.save(save_path)

        # Validate and preprocess audio using librosa
        try:
            y, sr = librosa.load(save_path, sr=None)  # Load audio file
            librosa.output.write_wav(save_path, y, sr)  # Save validated WAV file
        except Exception as e:
            return jsonify({"status": "error", "message": f"Audio validation failed: {str(e)}"}), 500

        return jsonify({"status": "success", "message": f"Audio saved and validated for question {question_id}."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def transcribe_audio(audio_path):
    """Transcribe audio file into text."""
    recognizer = sr.Recognizer()
    try:
        # Preprocess audio using librosa
        y, sample_rate = librosa.load(audio_path, sr=None)  # Load audio file
        cleaned_path = audio_path.replace('.wav', '_cleaned.wav')  # Save as cleaned file
        
        # Save the preprocessed audio using soundfile
        sf.write(cleaned_path, y, sample_rate, format='WAV') # Replace librosa.output.write_wav (deprecated)
        
        # Transcribe the cleaned audio
        with sr.AudioFile(cleaned_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            return transcription
    except Exception as e:
        return f"Error: {str(e)}"



def transcribe_all_audio():
    """Transcribe all audio files and save user answers."""
    recordings_dir = 'recordings'
    audio_user_answers = []
    
    # Load questions from CSV
    df = pd.read_csv('questions.csv', encoding='ISO-8859-1')
    questions = df[['qno', 'question']].to_dict('records')
    
    # Test the transcription with absolute path
    recordings_dir = r"C:\main project\mock_interview\recordings"

    # Iterate through all audio files in the recordings directory
    for file_name in os.listdir(recordings_dir):
        if file_name.endswith('.wav') and file_name.startswith('question_'):
            # Extract question number
            qno = int(file_name.split('_')[1].split('.')[0])
            audio_path = os.path.join(recordings_dir, file_name)
            
            # Transcribe the audio
            transcription = transcribe_audio(audio_path)
            
            # Get the corresponding question text
            question = next((q['question'] for q in questions if q['qno'] == qno), "Unknown Question")
            
            # Append result
            audio_user_answers.append({
                "qno": qno,
                "audio_user_answer": transcription,
                "question": question
            })
    
    # Sort the answers by `qno` to maintain alignment
    audio_user_answers = sorted(audio_user_answers, key=lambda x: x['qno'])

    # Save to a JSON file
    with open('audio_user_answers.json', 'w') as f:
        json.dump(audio_user_answers, f, indent=4)
    print("Transcriptions saved to audio_user_answers.json.")
    # Clear the recordings folder after processing all files
    clear_recordings_folder(recordings_dir)


# Function to clear recordings folder automatically
def clear_recordings_folder(folder_path='recordings'):
    """Delete all files in the specified folder."""
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
        print(f"All files in the '{folder_path}' folder have been deleted.")
    except Exception as e:
        print(f"Error while clearing folder: {str(e)}")


def analyze_results():
    """Analyze user answers and prepare question-wise results."""
    try:
        # Check if required files exist
        if not os.path.exists('audio_user_answers.json'):
            raise FileNotFoundError("audio_user_answers.json file not found.")
        if not os.path.exists('questions.csv'):
            raise FileNotFoundError("questions.csv file not found.")

        # Load user answers
        with open('audio_user_answers.json', 'r') as f:
            audio_user_answers = json.load(f)

        # Load correct answers
        df = pd.read_csv('questions.csv', encoding='ISO-8859-1')
        real_answers = df[['qno', 'Answer']].to_dict('records')

        # Ensure alignment by sorting both lists by `qno`
        real_answers = sorted(real_answers, key=lambda x: x['qno'])
        audio_user_answers = sorted(audio_user_answers, key=lambda x: x['qno'])

        results = []
        for audio_user_answer in audio_user_answers:
            qno = audio_user_answer.get('qno')
            user_text = audio_user_answer.get('audio_user_answer', '')

            # Find the correct answer for the question number
            correct_row = next((r for r in real_answers if r['qno'] == qno), None)
            if not correct_row:
                print(f"Warning: No correct answer found for question number {qno}")
                continue

            correct_text = correct_row['Answer']

            # Append analysis result
            results.append({
                "qno": qno,
                "question": audio_user_answer.get('question', ''),
                "audio_user_answer": user_text,
                "correct_answer": correct_text
            })

        # Save results
        with open('audio_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        print("Analysis complete. Results saved to audio_analysis_results.json.")
        return results

    except FileNotFoundError as e:
        print(f"File error: {e}")
        return []
    except Exception as e:
        print(f"Error in analyze_results: {e}")
        return []


@audio_interview.route('/audio_dash', methods=("POST", "GET"))
def audio_dash():
    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    # Instantiate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Directory containing all recordings
    recordings_dir = os.path.join('recordings')

    # Initialize overall results
    all_emotions = []  # Stores all emotions across all recordings
    question_emotion_distributions = {}  # Stores emotion distribution per question
    question_major_emotions = {}  # Stores major emotion per question

    # Iterate through each recording in the directory
    for recording_file in os.listdir(recordings_dir):
        # Full path to the recording file
        rec_sub_dir = os.path.join(recordings_dir, recording_file)

        # Predict emotion in the recording
        step = 1  # in seconds
        sample_rate = 16000  # in kHz
        emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step * sample_rate)

        # Add emotions to the overall list
        all_emotions.extend(emotions)

        # Calculate emotion distribution for this recording
        emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
        question_emotion_distributions[recording_file] = ', '.join([f'{emotion}: {dist}%' for emotion, dist in zip(SER._emotion.values(), emotion_dist)])

        # Determine the major emotion for this question
        major_emotion = max(set(emotions), key=emotions.count)
        question_major_emotions[recording_file] = major_emotion

    # Export all predicted emotions to .txt format
    SER.prediction_to_csv(all_emotions,timestamp, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')

    # Calculate overall emotion distribution
    overall_emotion_dist = [int(100 * all_emotions.count(emotion) / len(all_emotions)) for emotion in SER._emotion.values()]

    # Export overall emotion distribution to .csv format for D3JS
    df = pd.DataFrame(overall_emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db', 'audio_emotions_dist.txt'), sep=',')

     # Transcribe all audio files
    transcribe_all_audio()

    # Perform NLP-based analysis for question-wise results
    question_results = analyze_results()


    # Define emotion weights globally to avoid redundancy
    emotion_weights = {
    'Happy': 1.0,
    'Neutral': 0.8,
    'Surprise': 0.7,
    'Sad': 0.4,
    'Fear': 0.3,
    'Angry': 0.2,
    'Disgust': 0.1,
    }

    # Step 1: Compute weighted confidence score
    global emotion_data
    emotion_data = all_emotions

    total_emotion_weight = sum(emotion_weights.get(emotion, 0) for emotion in emotion_data)
    total_emotion_count = len(emotion_data)

    # Adjusted confidence score computation
    max_weight = max(emotion_weights.values())  # Max weight in the emotion weights
    confidence_score = (total_emotion_weight / (total_emotion_count * max_weight)) * 100 if total_emotion_count > 0 else 0

    # Step 2: Generate emotion frequency chart
    emotion_counts = Counter(emotion_data)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(
    emotion_weights.keys(),
    [emotion_counts.get(emotion, 0) for emotion in emotion_weights.keys()],
    color='skyblue'
    )

    # Add weighted contributions as labels
    for bar, emotion in zip(bars, emotion_weights.keys()):
        raw_count = emotion_counts.get(emotion, 0)
        weighted_score = emotion_weights[emotion] * raw_count
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, 
             f'{raw_count}\n({weighted_score:.2f})', ha='center')

    plt.title("Emotion Frequency and Weighted Contributions")
    plt.xlabel("Emotions")
    plt.ylabel("Frequency")
    plt.savefig('static/audio_emotion_chart.png')
    plt.close()

    # Step 3: Generate accuracy trend chart based on weighted confidence scores
    confident_emotions_per_frame = [emotion_weights.get(emotion, 0) for emotion in emotion_data]
    cumulative_weights = [sum(confident_emotions_per_frame[:i + 1]) for i in range(len(confident_emotions_per_frame))]
    accuracy_trend = [(cumulative_weights[i] / (i + 1)) * 100 for i in range(len(cumulative_weights))]

    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_trend, label="Confidence Accuracy Trend", color='red')
    plt.title("Accuracy Trend Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Weighted Confidence (%)")
    plt.legend()
    plt.savefig('static/audio_accuracy_trend.png')
    plt.close()

    # Step 4: Calculate dominant emotion and dynamic tips
    weighted_emotion_scores = {emotion: emotion_weights.get(emotion, 0) * count for emotion, count in emotion_counts.items()}
    dominant_emotion = max(weighted_emotion_scores, key=weighted_emotion_scores.get)

    # Dynamic tips based on the dominant emotion
    if dominant_emotion in ['Sad', 'Fear']:
        tips = "Try to stay relaxed and focused. Engage in mock interviews and use positive affirmations."
    elif dominant_emotion in ['Angry', 'Disgust']:
        tips = "Consider maintaining a calm demeanor. Practice measured responses to avoid appearing tense."
    elif dominant_emotion in ['Happy', 'Neutral', 'Surprise']:
        tips = "Excellent composure! Continue practicing to reinforce this confidence."
    else:
        tips = "Stay calm and review common interview scenarios to build confidence."

    print("All detected emotions:", all_emotions)
    print("Dominant emotion based on weights:", dominant_emotion)




    # Prepare data for rendering
    results_data = {
        'questions': question_results
    }

    # Render the updated template
    return render_template(
        'audio_dash.html',
        question_major_emotions=question_major_emotions,
        question_emotion_distributions=question_emotion_distributions,
        confidence=confidence_score,
        tips=tips,results=results_data,
    )

