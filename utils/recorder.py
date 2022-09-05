import json
import time
import os


def create_record(save_path, content: dict):
    assert os.path.exists(save_path), "Path {} Not Exist!".format(save_path)
    file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + ".json"
    # file_path = os.path.join(save_path, file_name)
    file_path = save_path + file_name
    print(file_path.center(20, "`"))
    json.dump(content, open(file_path, 'w'), indent=4)
    return file_path


def load_record(file_path):
    assert os.path.exists(file_path), "File {} Not Exist!".format(file_path)
    return json.load(open(file_path, 'r'))


def save_record(file_path, content):
    assert os.path.exists(file_path), "Path {} Not Exist!".format(file_path)
    json.dump(content, open(file_path, 'w'), indent=4)


if __name__ == '__main__':
    c = {"content": 1}
    create_record(save_path="./record/exam1/", content=c)
