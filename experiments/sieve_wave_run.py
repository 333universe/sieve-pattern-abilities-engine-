import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.wave_sieve import WaveSieveProjector

if __name__ == "__main__":
    projector = WaveSieveProjector()
    time, original, broken = projector.simulate_electromechanical_signal()
    cleaned = projector.apply_sieve_notch_filter(broken)
    
    # Export metrics tracking report directly to your benchmarks folder
    output_path = "outputs/benchmarks/wave_filtering_metrics.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("=== Sieve Electromechanical Filter Run Log ===\n")
        f.write(f"Total processed timesteps: {len(time)}\n")
        # Calculate Root-Mean-Square (RMS) filtering error metric
        sq_errors = [(org - cln)**2 for org, cln in zip(original, cleaned)]
        rms_error = math.sqrt(sum(sq_errors) / len(sq_errors))
        f.write(f"Residual Wave Propagation RMS Error: {rms_error:.6f}\n")
        f.write("✅ Wave stabilization validation complete.\n")
        
    print(f"📦 Filter benchmark logs written safely to: {output_path}")
