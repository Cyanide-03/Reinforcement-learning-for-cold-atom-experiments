'''
Created on Dec 10, 2024

@author: Kripali Jain
'''
from artiq.experiment import *

class DDS(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.setattr_device("urukul0_ch0")
        self.setattr_device("urukul0_ch1")
        self.setattr_device("urukul0_ch3")
        

    @kernel
    def run(self):
        self.core.reset()
        self.urukul0_ch0.init() #  AOM
        self.urukul0_ch0.set_att(4.3 *dB) #set the attenuation
        self.urukul0_ch0.set(110*MHz,0.0,0.5) # set the (frequency, phase, amplitude), amplitude

        self.urukul0_ch1.init() #  AOM
        self.urukul0_ch1.set_att(6.5 *dB) #set the attenuation
        self.urukul0_ch1.set(100*MHz,0.0,0.5) # set the (frequency, phase, amplitude), amplitude
        self.urukul0_ch1.sw.off()

        self.urukul0_ch3.init() # AOM
        self.urukul0_ch3.set_att(3.5 *dB) #set the attenuation
        self.urukul0_ch3.set(100*MHz,0.0,0.9) # set the (frequency, phase, amplitude), amplitude

        self.urukul0_ch0.sw.on()
        self.urukul0_ch1.sw.on()
        self.urukul0_ch3.sw.on()
        self.urukul0_ch3.sw.off()

