from flask import Flask,render_template
from video_interview import video_interview
from audio_interview import audio_interview

app = Flask(__name__)

# Register blueprints
app.register_blueprint(video_interview)
app.register_blueprint(audio_interview)


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)


