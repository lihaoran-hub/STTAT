import os

class Logger:
    def __init__(self, exp_name):
        if not os.path.exists('./logs/Metr-la/'):
            os.makedirs('./logs/Metr-la/')
            self.file = open('./logs/Metr-la/{}.log'.format(exp_name), 'w')
        else:
            self.file = open('./logs/Metr-la/{}.log'.format(exp_name), 'w')

    def log(self, content):
        self.file.write(content + '\n')
        self.file.flush()
