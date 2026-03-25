from artiq.experiment import *

class MOTExperiment(EnvExperiment):

    def build(self):

        self.setattr_device("core")
        self.setattr_device("dds")

        self.setattr_argument(
            "detuning",
            NumberValue(default=-20)
        )

    @kernel
    def run(self):

        self.core.reset()

        self.dds.set(
            frequency=self.detuning * MHz
        )

        delay(1*ms)

        # trigger camera