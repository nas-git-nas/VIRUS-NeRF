import time
import sys

def main():
    print("Start subprocess")
    time.sleep(5)
    print("End subprocess")
    sys.exit(1)

if __name__ == "__main__":
    main()