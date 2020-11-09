#External Script

from IQ_sample_based_sim_LoRa import *
from subprocess import call

class main_program(object):
    def __init__(self, path='provide the path of IQ_sample_based_sim_LoRa.py'):
        self.path=path
    
    def call_python_file(self):
        call(["python3","{}".format(self.path)])
    

if __name__=="__main__":
    c=main_program()
    for z in range(2): #no. of time want to run event script
        print("Event script simulation start",z)
        c.call_python_file()
   
