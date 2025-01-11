from flask import Flask, render_template, Response, jsonify, request,Blueprint
import pandas as pd
import cv2
import time
import librosa
import soundfile as sf
from deepface import DeepFace
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from collections import Counter
import json
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re
import os


video_interview = Blueprint('video_interview', __name__)


#VIDEO INTERVIEW


# Global variables to store emotion data
emotion_data = []
user_answers = []
questions_and_answers = []





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


def gen():
    global emotion_data
    video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0 for webcam
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Default emotion if no faces are detected
        emotion = "No face detected"
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis with error handling
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotion_data.append(emotion)

                # Draw rectangle and label around detected face
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            except Exception as e:
                print(f"Emotion detection error: {e}")

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

@video_interview.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_interview.route('/video-interview')
def video_interview_home():
    global questions_and_answers
    questions = load_questions_from_csv()
    selected_questions = questions[:10]

    # Save the selected questions globally or to a temporary file
    with open('selected_questions.json', 'w') as f:
        json.dump(selected_questions, f)

    return render_template('video_interview.html', questions=[q['question'] for q in selected_questions])


def load_selected_questions():
    try:
        with open('selected_questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    


@video_interview.route('/savevideo_audio', methods=["POST"])
def savevideo_audio():
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


def transcribe_videoaudio(audio_path):
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



def transcribe_all_videoaudio():
    """Transcribe all audio files and save user answers."""
    recordings_dir = 'recordings'
    user_answers = []
    
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
            transcription = transcribe_videoaudio(audio_path)
            
            # Get the corresponding question text
            question = next((q['question'] for q in questions if q['qno'] == qno), "Unknown Question")
            
            # Append result
            user_answers.append({
                "qno": qno,
                "user_answer": transcription,
                "question": question
            })
    
    # Sort the answers by `qno` to maintain alignment
    user_answers = sorted(user_answers, key=lambda x: x['qno'])

    # Save to a JSON file
    with open('user_answers.json', 'w') as f:
        json.dump(user_answers, f, indent=4)
    print("Transcriptions saved to user_answers.json.")


# Function to clear recordings folder automatically
def clear_recordings_videofolder(folder_path='recordings'):
    """Delete all files in the specified folder."""
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
        print(f"All files in the '{folder_path}' folder have been deleted.")
    except Exception as e:
        print(f"Error while clearing folder: {str(e)}")


def analyze_video_results():
    """Analyze user answers and prepare question-wise results."""
    try:
        # Check if required files exist
        if not os.path.exists('user_answers.json'):
            raise FileNotFoundError("user_answers.json file not found.")
        if not os.path.exists('questions.csv'):
            raise FileNotFoundError("questions.csv file not found.")

        # Load user answers
        with open('user_answers.json', 'r') as f:
            user_answers = json.load(f)

        # Load correct answers
        df = pd.read_csv('questions.csv', encoding='ISO-8859-1')
        real_answers = df[['qno', 'Answer']].to_dict('records')

        # Ensure alignment by sorting both lists by `qno`
        real_answers = sorted(real_answers, key=lambda x: x['qno'])
        user_answers = sorted(user_answers, key=lambda x: x['qno'])

        results = []
        for user_answer in user_answers:
            qno = user_answer.get('qno')
            user_text = user_answer.get('user_answer', '')

            # Find the correct answer for the question number
            correct_row = next((r for r in real_answers if r['qno'] == qno), None)
            if not correct_row:
                print(f"Warning: No correct answer found for question number {qno}")
                continue

            correct_text = correct_row['Answer']

            # Append analysis result
            results.append({
                "qno": qno,
                "question": user_answer.get('question', ''),
                "user_answer": user_text,
                "correct_answer": correct_text
            })

        # Save results
        with open('video_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        print("Analysis complete. Results saved to video_analysis_results.json.")
        return results

    except FileNotFoundError as e:
        print(f"File error: {e}")
        return []
    except Exception as e:
        print(f"Error in analyze_results: {e}")
        return []


@video_interview.route('/results')
def results():
    global emotion_data

    # Transcribe all audio files first
    transcribe_all_videoaudio()  # This will process all the audio files and save user answers
    
    # Count emotions and generate statistics
    emotion_counts = Counter(emotion_data)
    total_emotions = sum(emotion_counts.values())
    confidence_emotions = ['happy', 'neutral']
    confidence_score = sum(emotion_counts[emotion] for emotion in confidence_emotions) / total_emotions if total_emotions > 0 else 0

    # Perform NLP-based analysis for question-wise results
    question_results = analyze_video_results()

    # Generate emotion frequency chart
    plt.figure(figsize=(10, 5))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    plt.title("Emotion Frequency During Interview")
    plt.xlabel("Emotions")
    plt.ylabel("Frequency")
    plt.savefig('static/video_emotion_chart.png')
    plt.close()

# Generate accuracy trend chart
    confident_emotions_per_frame = [1 if emotion in confidence_emotions else 0 for emotion in emotion_data]
    accuracy_trend = [sum(confident_emotions_per_frame[:i+1]) / (i+1) * 100 for i in range(len(confident_emotions_per_frame))]

    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_trend, label="Confidence Accuracy Trend", color='orange')
    plt.title("Accuracy Trend Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig('static/video_accuracy_trend.png')
    plt.close()

    
    # Dynamic tips based on detected emotions
    if 'fear' in emotion_counts or 'sad' in emotion_counts:
        tips = "Try to stay relaxed and focused. Practice breathing techniques or mock interviews to reduce anxiety."
    elif 'angry' in emotion_counts or 'disgust' in emotion_counts:
        tips = "Try to maintain a positive outlook and calm demeanor. Focus on responding calmly to questions."
    elif 'happy' in emotion_counts or 'neutral' in emotion_counts:
        tips = "Great job on staying confident and composed! Keep up the good work."
    else:
        tips = "Keep calm and practice mock interviews to enhance your confidence."

    results_data = {
        'confidence': confidence_score*100,
        'tips': tips,
        'questions': question_results
    }

   # Clear recordings folder and global emotion data
    try:
        clear_recordings_videofolder('recordings')  # Clean up after everything is processed
        emotion_data.clear()
        print("Clearing files and emotion data after analysis is complete.")
    except Exception as e:
        print(f"Error while clearing recordings: {e}")

    return render_template('results.html', results=results_data, accuracy_chart='static/video_emotion_chart.png')
