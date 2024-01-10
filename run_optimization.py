import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import subprocess


def main():
    # run pso optimization
    cwd = os.getcwd()
    run_pso_path = os.path.join(cwd, "run_pso.py")
    # run_pso_path = os.path.join(cwd, "test_scripts/optimization", "test_run_pso.py")

    while True:
        try:
            print("running pso")
            exit_code = subprocess.call(["python3", run_pso_path])
        except:
            exit_code = 1

        print("exit code:", exit_code)

if __name__ == "__main__":
    main()