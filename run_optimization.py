import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import subprocess


def main():
    # run pso optimization
    cwd = os.getcwd()
    run_pso_path = os.path.join(cwd, "run_pso.py")
    # run_pso_path = os.path.join(cwd, "test_scripts/optimization", "test_run_pso.py")

    while True:
        exit_code = subprocess.call(["python", run_pso_path])

        if exit_code != 0:
            break

if __name__ == "__main__":
    main()