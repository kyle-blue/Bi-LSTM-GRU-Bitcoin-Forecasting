import dataset
import os

def main():
    dataset.start()

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    work_dir = os.path.normpath(cur_dir + "/../")
    os.environ["WORKSPACE"] = work_dir
    main()
    