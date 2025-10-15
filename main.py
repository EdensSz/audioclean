from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy.signal import butter, sosfilt, wiener
from scipy.ndimage import gaussian_filter1d
from scipy.signal import iirnotch
import os
import tempfile

app = FastAPI(title="Audio Cleaner API")

def ultra_clean_piano(input_path, output_path):
    """Pipeline optimisé pour netteté maximale"""
    
    print("🎹 Chargement de l'audio...")
    y, sr = librosa.load(input_path, sr=None, mono=False)
    
    # 1. NORMALISATION
    print("📊 Normalisation...")
    y = librosa.util.normalize(y)
    
    # 2. RÉDUCTION DE BRUIT
    print("🔇 Réduction de bruit...")
    y = nr.reduce_noise(
        y=y, 
        sr=sr, 
        prop_decrease=0.9,
        stationary=False,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50
    )
    
    # 3. RÉDUCTION DE RÉVERB
    print("💧 Réduction de réverb...")
    y = wiener(y, mysize=None)
    
    D = librosa.stft(y)
    D_magnitude = np.abs(D)
    D_phase = np.angle(D)
    
    decay_factor = 0.7
    for i in range(1, D_magnitude.shape[1]):
        D_magnitude[:, i] = np.maximum(
            D_magnitude[:, i], 
            D_magnitude[:, i-1] * decay_factor
        )
    
    D_clean = D_magnitude * np.exp(1j * D_phase)
    y = librosa.istft(D_clean)
    
    # 4. FILTRAGE FRÉQUENTIEL
    print("🎚️ Filtrage fréquentiel...")
    sos = butter(6, 60, btype='highpass', fs=sr, output='sos')
    y = sosfilt(sos, y)
    
    sos = butter(6, 8000, btype='lowpass', fs=sr, output='sos')
    y = sosfilt(sos, y)
    
    b, a = iirnotch(50, 30, sr)
    y = sosfilt([b], [a], y)
    
    # 5. ACCENTUATION DES ATTAQUES
    print("⚡ Accentuation des attaques...")
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=2.0)
    y = y_harmonic + (y_percussive * 1.5)
    
    # 6. GATE
    print("🚪 Application du gate...")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    gate_threshold = -45
    gate_mask = rms_db > gate_threshold
    gate_mask_interp = np.interp(
        np.arange(len(y)),
        np.arange(len(gate_mask)) * 512,
        gate_mask.astype(float)
    )
    
    gate_mask_smooth = gaussian_filter1d(gate_mask_interp, sigma=100)
    y = y * gate_mask_smooth
    
    # 7. TRIM SILENCE
    print("✂️ Trim du silence...")
    y, _ = librosa.effects.trim(y, top_db=35)
    
    # 8. CONVERSION MONO
    print("📻 Conversion mono...")
    if y.ndim > 1:
        y = librosa.to_mono(y)
    
    # 9. RESAMPLING
    print("🔄 Resampling à 22050 Hz...")
    target_sr = 22050
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # 10. NORMALISATION FINALE
    print("✨ Normalisation finale...")
    y = librosa.util.normalize(y)
    y = np.tanh(y * 1.2)
    y = librosa.util.normalize(y)
    
    # Sauvegarder
    print(f"💾 Sauvegarde dans {output_path}...")
    sf.write(output_path, y, sr)
    
    print("✅ Traitement terminé !")
    return output_path

@app.get("/")
def root():
    return {
        "message": "Audio Cleaner API 🎹",
        "version": "1.0.0",
        "endpoints": {
            "/clean": "POST - Upload audio file to clean",
            "/health": "GET - Check API status"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "audio-cleaner"}

@app.post("/clean")
async def clean_audio(file: UploadFile = File(...)):
    """
    Endpoint pour nettoyer un fichier audio
    """
    # Vérifier le type de fichier
    allowed_extensions = [".mp3", ".wav", ".m4a", ".ogg"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Extension non supportée. Utilisez: {', '.join(allowed_extensions)}"
        )
    
    # Créer fichiers temporaires
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    
    try:
        # Sauvegarder le fichier uploadé
        content = await file.read()
        temp_input.write(content)
        temp_input.close()
        
        print(f"📥 Fichier reçu: {file.filename} ({len(content)} bytes)")
        
        # Traiter l'audio
        ultra_clean_piano(temp_input.name, temp_output.name)
        
        # Renvoyer le fichier nettoyé
        return FileResponse(
            temp_output.name,
            media_type="audio/wav",
            filename=f"cleaned_{os.path.splitext(file.filename)[0]}.wav"
        )
    
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")
    
    finally:
        # Nettoyer les fichiers temporaires
        try:
            os.unlink(temp_input.name)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Démarrage du serveur sur le port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
