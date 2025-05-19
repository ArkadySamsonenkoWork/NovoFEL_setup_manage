import random
random.seed(42)

def PV(name: str):
    if name.startswith("MSC"):
        return SettingHangler(name)
    elif name.startswith("BPMS1"):
        return Handler(name)
    else:
        assert False, "wrong name of device"


class Handler:
    def __init__(self, name: str):
        self.name = name
        self._value = random.random()
        self._counter = 0
        self._random_coeff_1 = random.random()
        self._random_coeff_2 = random.random()

    def get(self):
        if self._counter < 100:
            self._value = self._value + random.random() / 100
        elif 40 <= self._counter < 80:
            self._value = 6
        else:
            self._counter = 0
        self._counter += 1
        return self._value

class SettingHangler(Handler):
    def put(self, value):
        self._value = value

    def get(self):
       self._value + self._value * random.random() / 10000
       return self._value