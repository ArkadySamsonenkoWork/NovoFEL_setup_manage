import random
random.seed(42)

def PV(name: str):
    if name.startswith("MSC"):
        return SettingHangler(name)
    elif name.startswith("BPMS"):
        return Handler(name)
    else:
        assert False, "wrong name of device"


class Handler:
    def __init__(self, name: str):
        self.name = name
        self._value = 0
        self._counter = 0

    def get(self):
        if self._counter < 100:
            self._value += random.random() / 5
        elif 100 <= self._counter < 200:
            pass
        else:
            self._counter = 0
        return self._value

class SettingHangler(Handler):
    def put(self, value):
        self._value = value