from artiq.experiment import *

class Hello(EnvExperiment):
    def build(self):
        pass

    def run(self):
        print("Hello ARTIQ")