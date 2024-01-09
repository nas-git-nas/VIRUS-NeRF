import time
import sys

def main():
    print("Start subprocess")
    time.sleep(1)
    print("End subprocess")
    sys.exit(0)

if __name__ == "__main__":
    main()