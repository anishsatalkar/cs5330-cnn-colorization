class State(object):
    def __init__(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, factor):
        self.val = val
        self.sum += val * factor
        self.count += factor
        self.average = self.sum / self.count



