import math
import struct
import os

class AudioSieveSequencer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        # Golden ratio conjugate for micro-rhythmic variations
        self.phi = (math.sqrt(5) - 1) / 2

    def generate_sine_wave(self, frequency, duration_ms, amplitude=0.3):
        """Generates a raw sine wave array for a specific note frequency."""
        total_samples = int((duration_ms / 1000.0) * self.sample_rate)
        wave = []
        for i in range(total_samples):
            # Calculate standard sound wave oscillation
            t = i / self.sample_rate
            value = amplitude * math.sin(2 * math.pi * frequency * t)
            wave.append(value)
        return wave

    def build_sieve_sequence(self, total_steps=64, bpm=120):
        """Generates a polyphonic 3-voice generative audio rhythm pattern."""
        step_duration_ms = int((60000 / bpm) / 4)  # 16th notes
        final_audio_stream = []

        # Define 3 distinct voice frequencies (Base, Mid, High accents)
        voice_frequencies = [110.0, 220.0, 440.0]

        print(f"🎵 Composing Sieve Waveforms across {total_steps} mathematical steps...")

        for step in range(total_steps):
            # Create a silent base buffer for this rhythmic step
            step_samples = int((step_duration_ms / 1000.0) * self.sample_rate)
            mixed_step_buffer = [0.0] * step_samples

            # Calculate sieve trigger states across 3 independent channels
            for voice_idx in range(3):
                # Irrational dial spacing prevents rhythmic collision or alignment loops
                dial = (step * (self.phi * (voice_idx + 1))) % 1.0
                trigger_threshold = 1.0 / (voice_idx + 3)

                if dial < trigger_threshold:
                    # Generate the tone trigger wave for this specific voice channel
                    tone = self.generate_sine_wave(voice_frequencies[voice_idx], step_duration_ms, amplitude=0.2)
                    
                    # Mix the wave into the concurrent multi-voice track
                    for i in range(min(len(mixed_step_buffer), len(tone))):
                        mixed_step_buffer[i] += tone[i]

            # Append the mixed polyphonic sample frame to the master audio stream
            final_audio_stream.extend(mixed_step_buffer)

        return final_audio_stream

    def save_wav_file(self, audio_data, output_filename="outputs/visuals/sieve_composition.wav"):
        """Writes the raw audio stream to a fully valid standard 16-bit WAV file."""
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        # Convert floating-point waves to standard 16-bit signed PCM binary structures
        binary_frames = bytearray()
        for sample in audio_data:
            # Clip tracking to prevent clipping distortion
            sample = max(-1.0, min(1.0, sample))
            integer_sample = int(sample * 32767)
            binary_frames.extend(struct.pack("<h", integer_sample))

        num_channels = 1
        bytes_per_sample = 2
        subchunk2_size = len(binary_frames)
        chunk_size = 36 + subchunk2_size
        byte_rate = self.sample_rate * num_channels * bytes_per_sample
        block_align = num_channels * bytes_per_sample

        # Build standard RIFF WAV structural header byte constraints
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
            num_channels, self.sample_rate, byte_rate, block_align, 16, b"data", subchunk2_size
        )

        with open(output_filename, "wb") as wav_file:
            wav_file.write(header)
            wav_file.write(binary_frames)

        print(f"✅ Generative audio composition saved successfully: {output_filename}")

if __name__ == "__main__":
    sequencer = AudioSieveSequencer()
    audio_stream = sequencer.build_sieve_sequence(total_steps=128, bpm=135)
    sequencer.save_wav_file(audio_stream)
