import datetime
import os
import time


class PrintLog:
    def __init__(self, path):
        self.path = path

    def println(self, message, end="\n"):
        print(message, end=end)

        time_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        file = open(self.path, mode="a")
        file.writelines(f"[{time_str}] {message}{end}")
        file.close()
