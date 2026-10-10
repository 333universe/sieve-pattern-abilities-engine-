import os
import sys
import subprocess

# 1. Force Python to see the root directory and its packages
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_script(folder, script_name):
    """Safely runs a script from the project root context."""
    script_path = os.path.join(ROOT_DIR, folder, script_name)
    print(f"\n🚀 Launching: {folder}/{script_name}...")
    
    try:
        # Runs the script while keeping the root directory as the execution base
        result = subprocess.run([sys.executable, script_path], cwd=ROOT_DIR, check=True)
        print(f"✅ Finished running {script_name} successfully!")
    except subprocess.CalledProcessError:
        print(f"❌ Error: {script_name} failed during execution.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file at {script_path}")

if __name__ == "__main__":
    print("=== Sieve Pattern Abilities Engine Launcher ===")
    print("1) Run Sieve Experiments  (sieve_experiments.py)")
    print("2) Run Sieve Explore      (sieve_explore.py)")
    print("3) Run Sieve Patterns     (sieve_patterns.py)")
    print("4) Exit")
    
    try:
        choice = input("\nEnter the number of the script you want to run: ").strip()
        
        if choice == "1":
            run_script("experiments", "sieve_experiments.py")
        elif choice == "2":
            run_script("experiments", "sieve_explore.py")
        elif choice == "3":
            run_script("experiments", "sieve_patterns.py")
        elif choice == "4" or choice.lower() == "exit":
            print("Goodbye!")
        else:
            print("Invalid choice. Please run the launcher again.")
            
    except KeyboardInterrupt:
        print("\nLauncher closed.")
