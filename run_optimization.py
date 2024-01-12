import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import subprocess


def main():
    # run pso optimization
    cwd = os.getcwd()
    run_pso_path = os.path.join(cwd, "run_pso.py")

    while True:
        print("running pso")
        exit_code = subprocess.call(["python3", run_pso_path])
        print("exit code:", exit_code)

        # print("exit code:", exit_code)
        # if exit_code != 0:
        #     break

if __name__ == "__main__":
    main()