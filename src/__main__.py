import app
import os

def main():
    app.start()

if __name__ == "__main__":
    os.environ["WORKSPACE"] = "/home/doidge/trading_sltm"
    main()