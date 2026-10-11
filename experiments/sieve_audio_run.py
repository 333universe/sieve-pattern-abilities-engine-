import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_sieve import AudioSieveSequencer

if __name__ == "__main__":
    sequencer = AudioSieveSequencer()
    # Generates a complex sequence across 128 micro-steps
    audio_stream = sequencer.build_sieve_sequence(total_steps=128, bpm=140)
    sequencer.save_wav_file(audio_stream, output_filename="outputs/visuals/sieve_composition.wav")

