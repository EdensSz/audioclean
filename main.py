import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy.signal import butter, sosfilt, wiener

def ultra_clean_piano(input_file, output_file):
    """
    Pipeline optimisé pour netteté maximale avant Basic Pitch
    """
    print("🎹 Chargement de l'audio...")
    y, sr = librosa.load(input_file, sr=None, mono=False)
    
    # 1. NORMALISATION
    print("📊 Normalisation...")
    y = librosa.util.normalize(y)
    
    # 2. RÉDUCTION DE BRUIT
    print("🔇 Réduction de bruit...")
    y = nr.reduce_noise(
        y=y, 
        sr=sr, 
        prop_decrease=0.9,  # Agressif pour TikTok
        stationary=False,   # Bruit non-stationnaire
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50
    )
    
    # 3. RÉDUCTION DE RÉVERB (séchage du son)
    print("💧 Réduction de réverb...")
    # Approche 1: Wiener filter (simple et efficace)
    y = wiener(y, mysize=None)
    
    # Approche 2: Compression temporelle (réduit la "queue" des notes)
    D = librosa.stft(y)
    D_magnitude = np.abs(D)
    D_phase = np.angle(D)
    
    # Réduire la décroissance temporelle (tail)
    decay_factor = 0.7  # Plus c'est bas, plus c'est sec (0.5-0.9)
    for i in range(1, D_magnitude.shape[1]):
        D_magnitude[:, i] = np.maximum(
            D_magnitude[:, i], 
            D_magnitude[:, i-1] * decay_factor
        )
    
    D_clean = D_magnitude * np.exp(1j * D_phase)
    y = librosa.istft(D_clean)
    
    # 4. FILTRAGE FRÉQUENTIEL (ne garder que piano)
    print("🎚️ Filtrage fréquentiel...")
    # High-pass: enlever rumble
    sos = butter(6, 60, btype='highpass', fs=sr, output='sos')
    y = sosfilt(sos, y)
    
    # Low-pass: enlever sifflement
    sos = butter(6, 8000, btype='lowpass', fs=sr, output='sos')
    y = sosfilt(sos, y)
    
    # Notch filter optionnel (enlever hum 50Hz/60Hz)
    from scipy.signal import iirnotch
    b, a = iirnotch(50, 30, sr)  # 50Hz (Europe) ou 60Hz (USA)
    y = sosfilt([b], [a], y)
    
    # 5. ACCENTUATION DES ATTAQUES
    print("⚡ Accentuation des attaques...")
    y_harmonic, y_percussive = librosa.effects.hpss(
        y, 
        margin=2.0  # Plus élevé = séparation plus nette
    )
    y = y_harmonic + (y_percussive * 1.5)  # Boost les attaques
    
    # 6. GATE (porte de bruit)
    print("🚪 Application du gate...")
    # Convertir en dB
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Gate threshold
    gate_threshold = -45  # dB (ajustable: -40 à -50)
    
    # Appliquer le gate avec interpolation pour éviter les clicks
    gate_mask = rms_db > gate_threshold
    gate_mask_interp = np.interp(
        np.arange(len(y)),
        np.arange(len(gate_mask)) * 512,
        gate_mask.astype(float)
    )
    
    # Smoothing du gate (évite les coupures brutales)
    from scipy.ndimage import gaussian_filter1d
    gate_mask_smooth = gaussian_filter1d(gate_mask_interp, sigma=100)
    
    y = y * gate_mask_smooth
    
    # 7. TRIM SILENCE
    print("✂️ Trim du silence...")
    y, _ = librosa.effects.trim(y, top_db=35)
    
    # 8. CONVERSION MONO
    print("📻 Conversion mono...")
    if y.ndim > 1:
        y = librosa.to_mono(y)
    
    # 9. RESAMPLING OPTIMAL
    print("🔄 Resampling à 22050 Hz...")
    target_sr = 22050
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # 10. NORMALISATION FINALE
    print("✨ Normalisation finale...")
    y = librosa.util.normalize(y)
    
    # Compression douce finale (optionnel, pour homogénéiser)
    y = np.tanh(y * 1.2)
    y = librosa.util.normalize(y)
    
    # Sauvegarder
    print(f"💾 Sauvegarde dans {output_file}...")
    sf.write(output_file, y, sr)
    
    print("✅ Traitement terminé !")
    return output_file

# UTILISATION
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
