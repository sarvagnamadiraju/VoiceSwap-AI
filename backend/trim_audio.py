import os
import sys
import librosa
import soundfile as sf

input_file = sys.argv[1]
output_file = sys.argv[2]
duration_sec = 45

print(f"Trimming {input_file} to {duration_sec} seconds...")
try:
    # Load only the first 45 seconds
    y, sr = librosa.load(input_file, sr=None, duration=duration_sec)
    sf.write(output_file, y, sr)
    print(f"Successfully saved to {output_file}")
except Exception as e:
    print(f"Error: {e}")
