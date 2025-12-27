import os
import io
import joblib
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
import shutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Optional transcription backends (whisper preferred, speech_recognition fallback)
WHISPER_MODEL = None
try:
    import whisper
    try:
        WHISPER_MODEL = whisper.load_model("small")
        print("✅ Whisper model loaded for transcription")
    except Exception:
        WHISPER_MODEL = None
        print("⚠️  Whisper installed but failed to load model")
except Exception:
    WHISPER_MODEL = None

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# VADER sentiment fallback
VADER_AVAILABLE = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_ANALYZER = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    VADER_ANALYZER = None
    VADER_AVAILABLE = False

# ---------------- PATH SETUP ----------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(APP_ROOT, "model")
FRONTEND_FOLDER = os.path.join(os.path.dirname(APP_ROOT), "frontend")

MODEL_PATH = os.path.join(MODEL_FOLDER, "mood_detection_model.pkl")
SCALER_PATH = os.path.join(MODEL_FOLDER, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_FOLDER, "label_encoder.pkl")

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    mood_encoder = joblib.load(ENCODER_PATH)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# Check FFmpeg availability
def check_ffmpeg():
    """Check if FFmpeg is available in system PATH"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

FFMPEG_AVAILABLE = check_ffmpeg()
if FFMPEG_AVAILABLE:
    print("✅ FFmpeg is available")
else:
    print("⚠️  WARNING: FFmpeg is not installed or not in PATH")

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(wav_path, sr_target=22050):
    """
    Extract 22 features matching the trained StandardScaler:
    - 13 MFCC coefficients
    - 9 additional acoustic features (spectral, temporal)
    = 22 features total
    """
    y, sr = librosa.load(wav_path, sr=sr_target)
    
    # 1. MFCCs (13 coefficients)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    
    # 2-9. Additional 9 acoustic features
    # Spectral centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Spectral rolloff
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=y))
    
    # Spectral flux
    stft = np.abs(librosa.stft(y))
    spectral_flux = np.mean(np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0)))
    
    # Temporal Centroid (average onset time in seconds)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    temporal_centroid = np.mean(librosa.frames_to_time(onset_frames, sr=sr)) if len(onset_frames) > 0 else 0
    
    # Chroma energy
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    
    # Spectral Contrast
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    # Zero crossing rate variance
    zcr_var = np.std(librosa.feature.zero_crossing_rate(y))
    
    # Combine all features: [13 MFCC + 9 additional = 22 features]
    features = np.hstack((mfcc, spectral_centroid, zcr, spectral_rolloff, 
                         rms, spectral_flux, temporal_centroid, chroma, 
                         spec_contrast, zcr_var))
    
    return features


def transcribe_audio(wav_path):
    """Transcribe audio to text using available backends.
    Returns empty string if no transcription backend is available or transcription fails.
    """
    # Whisper (local) preferred
    if WHISPER_MODEL is not None:
        try:
            print(f"[TRANSCRIBE] Using Whisper on {wav_path}")
            res = WHISPER_MODEL.transcribe(wav_path)
            text = res.get("text", "").strip()
            print(f"[TRANSCRIBE] Whisper result: '{text}'")
            return text
        except Exception as e:
            print(f"[TRANSCRIBE] Whisper error: {e}")
            pass

    # speech_recognition + Google as fallback (requires internet)
    if SR_AVAILABLE:
        try:
            print(f"[TRANSCRIBE] Using SpeechRecognition on {wav_path}")
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            print(f"[TRANSCRIBE] SpeechRecognition result: '{text}'")
            return text
        except Exception as e:
            print(f"[TRANSCRIBE] SpeechRecognition error: {e}")
            return ""

    print("[TRANSCRIBE] No transcription backend available")
    return ""


_TEXT_KEYWORD_MAP = {
    # Happy
    "happy": "Happy",
    "joy": "Happy",
    "joyful": "Happy",
    "great": "Happy",
    "wonderful": "Happy",
    "fantastic": "Happy",
    "amazing": "Happy",
    "good": "Happy",
    "excellent": "Happy",
    "love": "Happy",
    "glad": "Happy",
    "cheerful": "Happy",
    "delighted": "Happy",
    "blessed": "Happy",
    "awesome": "Happy",
    "cool": "Happy",
    "wonderful": "Happy",
    "brilliant": "Happy",
    
    # Excited
    "excited": "Excited",
    "exciting": "Excited",
    "thrilled": "Excited",
    "energized": "Excited",
    "pumped": "Excited",
    
    # Sad
    "sad": "Sad",
    "unhappy": "Sad",
    "depressed": "Sad",
    "depression": "Sad",
    "down": "Sad",
    "miserable": "Sad",
    "crying": "Sad",
    "cry": "Sad",
    "tears": "Sad",
    "lonely": "Sad",
    "alone": "Sad",
    "terrible": "Sad",
    "awful": "Sad",
    "bad": "Sad",
    "worst": "Sad",
    "hate": "Sad",
    "disappointed": "Sad",
    "disappointed": "Sad",
    
    # Angry
    "angry": "Angry",
    "mad": "Angry",
    "furious": "Angry",
    "rage": "Angry",
    "upset": "Angry",
    "irritated": "Angry",
    "annoyed": "Angry",
    "frustrated": "Angry",
    "pissed": "Angry",
    
    # Confused
    "confused": "Confused",
    "confusion": "Confused",
    "unclear": "Confused",
    "puzzled": "Confused",
    "lost": "Confused",
    "bewildered": "Confused",
    "perplexed": "Confused",
    
    # Calm
    "calm": "Calm",
    "peaceful": "Calm",
    "relaxed": "Calm",
    "serene": "Calm",
    "quiet": "Calm",
    
    # Neutral
    "neutral": "Neutral",
    "okay": "Neutral",
    "fine": "Neutral",
    "meh": "Neutral",
    "alright": "Neutral",
    
    # Fearful
    "afraid": "Fearful",
    "scared": "Fearful",
    "fear": "Fearful",
    "terror": "Fearful",
    "panic": "Fearful",
    "anxious": "Fearful",
    
    # Surprised
    "surprised": "Surprised",
    "surprise": "Surprised",
    "shocked": "Surprised",
    "wow": "Surprised",
    "astonished": "Surprised",
    
    # Bored
    "bored": "Bored",
    "boring": "Bored",
    "tedious": "Bored",
    "dull": "Bored",
}


def text_emotion_from_keywords(text):
    if not text:
        return None
    t = text.lower()
    counts = {}
    for kw, label in _TEXT_KEYWORD_MAP.items():
        if kw in t:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def text_emotion_from_text(text):
    """Return a coarse emotion label from text using keywords first,
    then VADER sentiment as a fallback (if available).
    """
    if not text:
        return None
    # 1) keywords (most reliable)
    k = text_emotion_from_keywords(text)
    if k is not None:
        print(f"[TEXT_EMOTION] Matched keyword '{k}' from text: '{text}'")
        return k

    # 2) VADER sentiment (lower thresholds for better sensitivity)
    if VADER_AVAILABLE and VADER_ANALYZER is not None:
        scores = VADER_ANALYZER.polarity_scores(text)
        comp = scores.get("compound", 0.0)
        print(f"[TEXT_EMOTION] VADER compound score: {comp} for text: '{text}'")
        
        # Lowered thresholds to be more responsive
        if comp >= 0.1:
            print(f"[TEXT_EMOTION] VADER -> Happy (comp={comp})")
            return "Happy"
        if comp <= -0.1:
            print(f"[TEXT_EMOTION] VADER -> Sad (comp={comp})")
            return "Sad"
        return "Neutral"

    print(f"[TEXT_EMOTION] No match for text: '{text}'")
    return None

# ---------------- FLASK APP ----------------
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path="")
CORS(app)

@app.route("/")
def home():
    return send_from_directory(FRONTEND_FOLDER, "index.html")

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory(os.path.join(FRONTEND_FOLDER, "static"), path)

# ---------------- PREDICT ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    if not FFMPEG_AVAILABLE:
        return jsonify({
            "error": "FFmpeg is not installed. Please install FFmpeg:\n"
                    "Windows: Download from https://ffmpeg.org/download.html\n"
                    "Mac: brew install ffmpeg\n"
                    "Linux: sudo apt-get install ffmpeg\n"
                    "Then add FFmpeg to your system PATH and restart the server."
        }), 500

    audio_file = request.files["file"]

    try:
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
            audio_file.save(webm_file.name)
            webm_path = webm_file.name

        wav_path = webm_path.replace(".webm", ".wav")

        # Convert WEBM → WAV
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", webm_path, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30
            )
        except FileNotFoundError:
            os.remove(webm_path)
            return jsonify({
                "error": "FFmpeg conversion failed. FFmpeg not found in PATH.\n"
                        "Install FFmpeg from: https://ffmpeg.org/download.html"
            }), 500
        except subprocess.TimeoutExpired:
            os.remove(webm_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return jsonify({"error": "Audio conversion timeout"}), 500

        # Check if conversion was successful
        if not os.path.exists(wav_path):
            os.remove(webm_path)
            return jsonify({"error": "Audio conversion failed. FFmpeg may not be properly installed."}), 500

        # Extract features
        try:
            features = extract_features(wav_path)
        except Exception as e:
            os.remove(webm_path)
            os.remove(wav_path)
            return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500

        features_scaled = scaler.transform([features])

        # Predict (audio-only)
        pred_encoded = model.predict(features_scaled)
        audio_emotion = mood_encoder.inverse_transform(pred_encoded)[0]

        # Transcribe audio (if backend available) and derive text-based emotion
        transcription = transcribe_audio(wav_path)
        text_emotion = text_emotion_from_text(transcription)

        # Decide final emotion: prefer text emotion if available (keywords or sentiment)
        final_emotion = text_emotion if text_emotion is not None else audio_emotion

        # Debug logging to help diagnose why overrides aren't applied
        print("\n" + "="*70)
        print("[PREDICT RESULT]")
        print(f"  Audio transcription: '{transcription}'")
        print(f"  Audio-only prediction: {audio_emotion}")
        print(f"  Text-based emotion: {text_emotion}")
        print(f"  Final emotion (text override audio): {final_emotion}")
        print(f"  Used text override: {text_emotion is not None}")
        print("="*70 + "\n")

        # Cleanup temp files
        try:
            os.remove(webm_path)
        except Exception:
            pass
        try:
            os.remove(wav_path)
        except Exception:
            pass

        return jsonify({
            "emotion": final_emotion,
            "audio_emotion": audio_emotion,
            "transcription": transcription,
            "used_text_override": text_emotion is not None
        })

    except Exception as e:
        print("BACKEND ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)