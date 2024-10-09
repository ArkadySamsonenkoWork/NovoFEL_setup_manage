class Some:
    def __repr__(self):
        return "123"

    def __str__(self):
        return "234"


a = Some()
print(repr(a))