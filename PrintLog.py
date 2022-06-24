import datetime
import os
import time


class PrintLog:
    def __init__(self, path):
        self.path = path

    def println(self, message, end="\n", log=True):
        print(message, end=end)

        if log:
            self.log(message, end)

    def log(self, message, end="\n"):
        time_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        file = open(self.path, mode="a")
        file.writelines(f"[{time_str}] {message}{end}")
        file.close()
