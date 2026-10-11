import math
import os

class QuaternionSieveStabilizer:
    def __init__(self):
        # Golden ratio conjugate for non-repeating orbital phase advances
        self.phi = (math.sqrt(5) - 1) / 2
        # Plastic ratio conjugate for cross-axial orthogonal decoupling
        self.psi = 0.7548776662466927

    def normalize_quaternion(self, w, x, y, z):
        """Ensures the quaternion sits perfectly on a 4D unit sphere (valid rotation)."""
        magnitude = math.sqrt(w**2 + x**2 + y**2 + z**2)
        if magnitude == 0:
            return 1.0, 0.0, 0.0, 0.0
        return w / magnitude, x / magnitude, y / magnitude, z / magnitude

    def generate_spinor_rotation(self, angle, axis_x, axis_y, axis_z):
        """Converts an angle and an axis into an orientation Quaternion Spinor."""
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)
        
        # Normalize the structural rotation axis
        axis_mag = math.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
        ax = axis_x / axis_mag if axis_mag > 0 else 0.0
        ay = axis_y / axis_mag if axis_mag > 0 else 0.0
        az = axis_z / axis_mag if axis_mag > 0 else 0.0
        
        w = math.cos(half_angle)
        x = ax * sin_half
        y = ay * sin_half
        z = az * sin_half
        return self.normalize_quaternion(w, x, y, z)

    def simulate_stabilization_pass(self, total_frames=1000):
        """Simulates 3D spatial drift and evaluates sieve alignment metrics."""
        print("🌀 Initializing 4-Dimensional Quaternion Sieve Spinors...")
        
        drift_log = []
        stabilized_log = []
        
        for n in range(1, total_frames + 1):
            # 1. Simulate complex, multi-axis rotational jitter/drift
            # Irrational steps generate a realistic wandering orientation matrix
            jitter_angle = 0.15 * math.sin(n * self.phi)
            dx = math.cos(n * self.psi)
            dy = math.sin(n * self.phi)
            dz = math.cos(n * (self.phi + self.psi))
            
            # Target corrupted drift quaternion
            qw, qx, qy, qz = self.generate_spinor_rotation(jitter_angle, dx, dy, dz)
            drift_log.append((qw, qx, qy, qz))
            
            # 2. Apply Sieve Projector to calculate counter-rotational stability
            # Evaluate phase matrix boundaries to compute the corrective quaternion
            phase = (n * self.phi) % 1.0
            if 0.25 < phase < 0.75:
                # Active cancellation via conjugate spinor inversion
                sqw, sqx, sqy, sqz = self.generate_spinor_rotation(-jitter_angle, dx, dy, dz)
            else:
                sqw, sqx, sqy, sqz = 1.0, 0.0, 0.0, 0.0 # Maintain neutral baseline identity
                
            stabilized_log.append((sqw, sqx, sqy, sqz))
            
        return drift_log, stabilized_log

if __name__ == "__main__":
    stabilizer = QuaternionSieveStabilizer()
    drift, stabilized = stabilizer.simulate_stabilization_pass()
    print(f"✅ Processed {len(drift)} 4D orientation frames successfully.")
