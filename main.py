from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy.signal import butter, sosfilt, lfilter, iirnotch
from scipy.ndimage import gaussian_filter1d
import os
import tempfile

app = FastAPI(title="Audio Cleaner API")

def ultra_clean_piano(input_path, output_path):
    """Pipeline optimisé pour netteté maximale - VERSION RAPIDE"""
    
    print("🎹 Chargement de l'audio...")
    # Charger directement en mono à 22050 Hz pour gagner du temps
    y, sr = librosa.load(input_path, sr=22050, mono=True)
    
    print(f"⏱️ Durée audio: {len(y)/sr:.2f}s")
    
    # 1. NORMALISATION
    print("📊 Normalisation...")
    y = librosa.util.normalize(y)
    
    # 2. RÉDUCTION DE BRUIT (version plus rapide)
    print("🔇 Réduction de bruit...")
    y = nr.reduce_noise(
        y=y, 
        sr=sr, 
        prop_decrease=0.85,  # Moins agressif = plus rapide
        stationary=True,     # Plus rapide que False
        n_fft=1024          # Réduit pour vitesse
    )
    
    # 3. FILTRAGE FRÉQUENTIEL
    print("🎚️ Filtrage fréquentiel...")
    # High-pass: enlever rumble
    sos = butter(4, 80, btype='highpass', fs=sr, output='sos')  # Ordre réduit de 6 à 4
    y = sosfilt(sos, y)
    
    # Low-pass: enlever sifflement
    sos = butter(4, 7000, btype='lowpass', fs=sr, output='sos')
    y = sosfilt(sos, y)
    
    # 4. ACCENTUATION DES ATTAQUES (version simplifiée)
    print("⚡ Accentuation des attaques...")
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=1.5)  # margin réduit = plus rapide
    y = y_harmonic + (y_percussive * 1.3)
    
    # 5. GATE SIMPLIFIÉ (plus rapide)
    print("🚪 Application du gate...")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    gate_threshold = -40  # Moins agressif
    gate_mask = rms_db > gate_threshold
    gate_mask_interp = np.interp(
        np.arange(len(y)),
        np.arange(len(gate_mask)) * 512,
        gate_mask.astype(float)
    )
    
    # Smoothing plus simple
    gate_mask_smooth = gaussian_filter1d(gate_mask_interp, sigma=50)  # sigma réduit
    y = y * gate_mask_smooth
    
    # 6. TRIM SILENCE
    print("✂️ Trim du silence...")
    y, _ = librosa.effects.trim(y, top_db=30)
    
    # 7. NORMALISATION FINALE
    print("✨ Normalisation finale...")
    y = librosa.util.normalize(y)
    y = np.tanh(y * 1.1)
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
        "version": "2.0.0 - Fast",
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
    import time
    start_time = time.time()
    
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
        
        elapsed = time.time() - start_time
        print(f"⏱️ Temps de traitement: {elapsed:.2f}s")
        
        # Renvoyer le fichier nettoyé
        return FileResponse(
            temp_output.name,
            media_type="audio/wav",
            filename=f"cleaned_{os.path.splitext(file.filename)[0]}.wav"
        )
    
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
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
