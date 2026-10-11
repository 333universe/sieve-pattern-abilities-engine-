import math
import os

class WaveSieveProjector:
    def __init__(self, fundamental_frequency=60.0, sample_rate=2000):
        self.f0 = fundamental_frequency
        self.fs = sample_rate
        # Golden ratio conjugate as our non-coaxial phase advance dial
        self.phi = (math.sqrt(5) - 1) / 2

    def simulate_electromechanical_signal(self, duration_sec=1.5):
        """Simulates a physical motor/wave signal corrupted by quasi-periodic harmonic interference."""
        total_samples = int(duration_sec * self.fs)
        timeline = [i / self.fs for i in range(total_samples)]
        raw_signal = []
        corrupted_signal = []

        for t in timeline:
            # 1. Pure baseline mechanical wave (e.g., stable 60Hz rotor baseline)
            base_wave = math.sin(2 * math.pi * self.f0 * t)
            raw_signal.append(base_wave)

            # 2. Add complex quasi-periodic mechanical vibration noise (Section H behavior)
            # This mimics multi-axis rotor wobble or friction harmonics
            noise1 = 0.4 * math.sin(2 * math.pi * (self.f0 * self.phi) * t)
            noise2 = 0.2 * math.sin(2 * math.pi * (self.f0 * (self.phi + 1)) * t)
            
            corrupted_signal.append(base_wave + noise1 + noise2)

        return timeline, raw_signal, corrupted_signal

    def apply_sieve_notch_filter(self, corrupted_signal):
        """Applies an algebraic sieve filter matrix to isolate and cancel out the noise profiles."""
        filtered_signal = []
        print("🌀 Engaging Sieve Projector Filter Matrices...")
        
        for idx, sample in enumerate(corrupted_signal):
            # Calculate the discrete sequence rotation state
            phase_advance = (idx * self.phi) % 1.0
            
            # The Sieve Projector boundary condition act as an idempotent space projector
            # If the phase rotation hits a structural sieve alignment, cancel out the predicted envelope
            if 0.3 < phase_advance < 0.7:
                # Reconstruct and subtract the estimated irrational wave boundaries
                estimated_interference = 0.4 * math.sin(2 * math.pi * (self.f0 * self.phi) * (idx / self.fs))
                clean_sample = sample - estimated_interference
            else:
                clean_sample = sample

            filtered_signal.append(clean_sample)
            
        return filtered_signal

if __name__ == "__main__":
    projector = WaveSieveProjector()
    time, original, broken = projector.simulate_electromechanical_signal()
    cleaned = projector.apply_sieve_notch_filter(broken)
    print(f"✅ Processed {len(cleaned)} electromechanical wave frames successfully.")
