import subprocess
import os

def main():
    print("Start main")

    # run pso optimization
    cwd = os.getcwd()
    # run_pso_path = os.path.join(cwd, "optimization", "run_pso.py")
    run_pso_path = os.path.join(cwd, "test_scripts/optimization", "test_run_pso.py")


    exit_code = subprocess.call(["python", run_pso_path])
    print(f"exit_code={exit_code}")

    print("End main")

if __name__ == "__main__":
    main()