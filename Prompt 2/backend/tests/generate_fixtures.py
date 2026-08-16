import math
import os
import struct
import wave
from pathlib import Path


def create_pcm_wav(filepath: Path, duration_sec: float = 2.5, sample_rate: int = 16000, freq: float = 440.0, volume: float = 0.5):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    
    with wave.open(str(filepath), "w") as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)      # 16-bit PCM (2 bytes per sample)
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            t = i / sample_rate
            # Generate modulated sine wave tone simulating speech energy envelope
            envelope = math.sin(math.pi * t / duration_sec)
            sample_val = int(volume * envelope * 32767.0 * math.sin(2.0 * math.pi * freq * t))
            wav_file.writeframes(struct.pack("<h", sample_val))


def generate_all_fixtures():
    fixtures_dir = Path("backend/tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "eng_query.wav": (2.0, 440.0, 0.6),
        "hin_query.wav": (2.5, 350.0, 0.6),
        "noisy_mumbled.wav": (1.8, 120.0, 0.15),
        "weather_offtopic.wav": (2.2, 520.0, 0.55),
        "short_sample.wav": (1.0, 400.0, 0.5),
    }

    for fname, (duration, freq, vol) in fixtures.items():
        fpath = fixtures_dir / fname
        create_pcm_wav(fpath, duration_sec=duration, freq=freq, volume=vol)
        print(f"Generated fixture: {fpath} ({fpath.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_all_fixtures()
