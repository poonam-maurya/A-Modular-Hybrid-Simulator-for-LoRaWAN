
#the access probabilities will be determined based on the slots
#the channel "attempt rate" will be determined by the sample interval as well

from bitstring import BitArray, BitStream
import math
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import geom
from scipy.fftpack import fft
import os
import xlsxwriter 
NumGW = 1; #no. of gateway
NumDev=3; #no. of mobile devices
PacketLenBytes=40;
#workbook = xlsxwriter.Workbook('hello.xlsx')
#worksheet = workbook.add_worksheet()
theta_m=30;
row=0
col=0
target_thinning_prob=.8
mobilityscale=15000000
a_i=0
a_up=[]
number_of_preamble=7
preamble_up=[]
I_Q_sample_physical_layer_preamble=[]  

log_enabled = {}
log_enabled["NODE"]=0
log_enabled["GW"]=0
log_enabled["MAIN"]=1

def print_log(device,*argv):
    if log_enabled[device]==1:
        print(device,end ="")
        print(":\t",end=" ")
        for arg in argv:
            print(arg,end =" ")
        print()

def save_context(varname,varvalue):
    filename="SavedVars/"+varname
    f=open(filename, "w")
    f.write(varvalue)
    f.close()

def load_context(varname,defaultvalue):
    filename="SavedVars/"+varname
    if os.path.exists(filename):
        f=open(filename, "r")
        return(f.read())
        f.close()
    else:
        return(defaultvalue)

def MAC_PHYSICAL_LAYER_PACKET(mac_payload_size,SF,mac_payload_stream=None):
    if mac_payload_stream==None:
        mac_payload_stream = BitArray(mac_payload_size) ##ba change## #generate bitstream of length mac_payload_size
    #chopping the mac bit-stream into packets of SF length for LoRa modulation 
    step=0
    array_physical_symbol_bit=[]
    array_physical_symbol_decimal=[]
    I_Q_sample_physical_layer=[]
    M=2**SF
    for i in range(int(mac_payload_size/SF)):
        array_physical_symbol_bit.append(mac_payload_stream[step:step+int(SF)])   
        step=int(SF)+step

    #converting the each pysical layer packet bit-stream into its decimal equivalent 
    for j in range(len(array_physical_symbol_bit)):
        i=0
        for bit in array_physical_symbol_bit[j]:
            i=(i<<1) |bit
        array_physical_symbol_decimal.append(i)
        
    # modulating each physical packet symbol with up-chrips
    a_up=array_physical_symbol_decimal
    #preamble aadition in mac payload at physical layer in order to send in air
    for i in range(number_of_preamble):
        for n in range(int(M)):
            preamble_up.append(np.exp(1j*2*np.pi*(((n**2)/(2*M))+((a_i/M)-.5)*n)))
      
    for i in range(len(a_up)): #for each symbol
        Lora_signal_up1=[]
        for n in range(int(M)):
            Lora_signal_up1.append(np.exp(1j*2*np.pi*(((n**2)/(2*M))+((a_up[i]/M)-.5)*(n))))
            I_Q_sample_physical_layer.append(np.exp(1j*2*np.pi*(((n**2)/(2*M))+((a_up[i]/M)-.5)*(n)))) #collecting total I/Q samples of physical layer packet
    
    I_Q_sample_physical_layer_preamble.append(preamble_up+I_Q_sample_physical_layer)

    return I_Q_sample_physical_layer

def LoRa_Receiver_demodulation(I_Q_sample_physical_layer,SF):
    Received_packet_IQ=[]
    Lora_up_conjugate1=[]
    step1=0
    a_i=0
    M=2**SF
    received_symbol=[]
    received_symbol_bits=[]
    received_symbol_bits1=[]
    received_symbol_bits2=[]
    mac_payload_at_receiver=[]

    for i in range(int(len(I_Q_sample_physical_layer)/(M))):
        Received_packet_IQ.append(I_Q_sample_physical_layer[step1:step1+int(M)])
        step1=step1+int(M)
    for i in range(len(Received_packet_IQ)):
        dechriping_lora_up1=[]
        for n in range(int(M)):
            Lora_up_conjugate1.append(np.exp(-1j*2*np.pi*(((n**2)/(2*M))+((a_i/M)-.5)*n)))
            dechriping_lora_up1.append(Received_packet_IQ[i][n]*Lora_up_conjugate1[n])
        d_fft=fft(dechriping_lora_up1)
        maximum_fre=np.argmax(d_fft)
        received_symbol.append(maximum_fre)

    for i in range(len(received_symbol)):
        received_symbol_bits.append(bin(received_symbol[i]))
        received_symbol_bits1.append(received_symbol_bits[i][2:])
        received_symbol_bits2.append(received_symbol_bits1[i].zfill(int(SF)))
    mac_payload_at_receiver.append("".join(received_symbol_bits2))
    
    return received_symbol


def collision_detection(num_recived_sample, collision_status): 
    if num_recived_sample>1:
        if collision_status==0:
            collision_status=1
            print("collision detected at gateway")
    return(collision_status)
    
    
def transmission_parameter():
    SF = np.random.randint(7,12)
    return(SF)

def application_payload_format(tx_symbol,num,SF):    #P: grneration of application payload
    #print("transmitting symbol",tx_symbol,num,SF)
    payload= BitArray(int=tx_symbol,length=SF)
    payload=payload+BitArray(int=num,length=SF)
    if num==0:
        print("transmitting symbol*******",tx_symbol,num,SF)
    return payload
r=3
location_node=[]
def node_distribution():
    location=int(np.ceil(np.random.randint(0,359)/theta_m))
    X = r * math.cos(location*30)  
    Y = r * math.sin(location*30)  
    location_node.append((X,Y))
    if len(location_node)==NumDev:
        xs=[x[0] for x in location_node]
        ys=[x[1] for x in location_node]
        # plt.plot(xs, ys, color='green', linestyle='dashed', linewidth = 3, 
        #  marker='o', markerfacecolor='blue', markersize=12)
        # dy = 0.1
        # fig, ax = plt.subplots()
        # plt.errorbar(xs, ys, dy, fmt='.k')
        plt.scatter(xs,ys)
        plt.show()
        
    return(location)

def next_node_location(time,loc):
    for i in range(time//mobilityscale):#get time/mobilityscale number of transitions
            if np.random.random()<0.5:
                loc=(loc + 1)%int(360/theta_m)
            else:
                loc=(loc - 1)%int(360/theta_m)
    return(loc)
    
    
class Node():
    #initializes the location and other parameters of the node
    def __init__(self,space="ring",mobility="SRW",num=1):#for symmetric random walk
        strn="node"+str(num)+"loc"
        self.loc=int(load_context(strn,node_distribution())); #initial angle, in theta_m units
        #print_log("NODE","Initial Location ",self.loc);
        self.mobilityscale=15000000; #mobilityscale is in terms of samples. For each mobilityscale number of samples, the node moves left or right with equal probability
        #this is also the scale at which next transmission probabilities are decided
        strn="node"+str(num)+"p"
        #self.p=float(load_context(strn,np.random.uniform(target_thinning_prob/2.0,target_thinning_prob))); #probability of transmitting in a sample duration
        self.p=float(load_context(strn,np.random.uniform(0.1,0.4))); #probability of transmitting in a sample duration
        #the above initialization should be less than the target thinning probability as target thinning probability is upper bounded by p
        strn="node"+str(num)+"next_event"
        self.next_event=int(load_context(strn,self.mobilityscale*(np.random.uniform(1,1.05))*geom.rvs(self.p))); #gets the first value for tranmission slot. staggers the exact transnmission slot to avoid inter-node synchronization
        #this is not the global time. this is time-to-next-event
        self.state="IDLE"; 
        self.samplenum=0;  #the ongoing IQ sample number
        self.num=num;
        strn="node"+str(num)+"num_attempts"
        self.num_attempts=int(load_context(strn,1));
        #2 is added to the length to ensure that the begining and end
        #are zero so that the receiver can perform energy detection.
        strn="node"+str(num)+"SF"
        self.SF=int(load_context(strn,transmission_parameter()))
        payload= application_payload_format(self.loc,self.num,self.SF)
        
        
        y=MAC_PHYSICAL_LAYER_PACKET(mac_payload_size=len(payload),SF=self.SF,mac_payload_stream=payload)
        
        self.pktlen=len(y)+2; #assume len(y) IQ samples per physical layer transmission.
        self.IQ=(0+0j)*np.ones(self.pktlen); #replace this by IQ samples
        
        self.IQ[1:len(y)+1]=y;
        strn="node"+str(num)+"last_event_time"
        self.last_event_time=int(load_context(strn,0));
        #print_log("NODE","Initial next event schedule",self.last_event_time+self.next_event);
        
    def get_node_num(self):
        return self.num

    def get_next_time(self):
        return self.next_event
    
    def do_event(self):
        self.change_loc=next_node_location(self.next_event,self.loc); #self.next_event is the last time interval
        self.loc=self.change_loc
        self.last_event_time=self.last_event_time+self.next_event;#current time
        #print("last event time of node**********",self.last_event_time)
        if self.state=="IDLE": #next step is transmission
            self.state="Tx";
            self.samplenum=1;
            print_log("NODE", "attempt no. ",self.num,self.num_attempts,self.loc,self.last_event_time)
            self.next_event=1; #next event is IQ sample transmission again
        else:
            if self.state=="Tx":
                if self.samplenum==self.pktlen: #last packet
                    
                    self.state="IDLE"; 
                    self.next_event=int(self.mobilityscale*(np.random.uniform(1,1.05))*geom.rvs(self.p)); #gets the first value for tranmission slot. staggers the exact transnmission slot to avoid inter-node synchronization
                    
                    self.cur_loc=self.get_loc()
                    print_log("NODE", "Going to Idle...",self.num,self.last_event_time,self.cur_loc);
                    self.change_loc=next_node_location(self.next_event,self.loc);
                    self.loc=self.change_loc
                    payload = application_payload_format(self.loc,self.num,self.SF)
                    
                    y=MAC_PHYSICAL_LAYER_PACKET(mac_payload_size=len(payload),SF=self.SF,mac_payload_stream=payload)
                    self.IQ[1:len(y)+1]=y;
                    self.samplenum=0;
                    
                    self.num_attempts=self.num_attempts+1;
                    
                else: #not transiting to IDLE
                    self.state="Tx";
                    self.samplenum=self.samplenum+1;
                    self.next_event=1; #next event is IQ sample transmission again

    def get_state(self):
        return self.state;

    def get_samplenum(self):
        return self.samplenum;

    def get_iq(self,num):
        if num<self.pktlen:
            return self.IQ[num];
        else:
            return 0+0j; #nothing to be sent when going to idle. this should never happen

    def get_pktlen(self):
        return(self.pktlen);

    def get_loc(self):
        return(self.loc);
    def get_SF(self):
        return(self.SF)

    

    def get_last_event_time(self):
        return(self.last_event_time)
    
class GW():
    #initializes the location and other parameters of the node
    def __init__(self):#for symmetric random walk
        strn="gateway_loc"
        self.loc=int(load_context(strn,np.random.randint(0,359))); #Fixed Locations
        self.iq=[];
        self.rx=[];
        self.energy_threshold=0.5;# the energy threshold for detection
        self.frame_ongoing=0; #to differentiate start of frame from end of frame
        self.current_iq_train=[];
        self.is_collision=0;
        self.decoded=0;
        self.was_collision=0;
        self.node_sample_count=1000000;
        self.node_num=55 #just initialise 
        self.num_sample_current_instant=0
        self.received_SF=0
    def start_receiving_iq(self): #means a new event has happened
        self.num_sample_current_instant=0;#reset for the next sample
        self.iq.append(0+0j);
        #print_log("GW", "initialized iq",self.iq);

    def receive_iq(self,loc,source_iq,node,SF): #add iq component to the currently received sample
        #loc is the location of the sender node. this is to get the channel
        self.received_SF=SF
        self.node_num=node
        
        if abs(source_iq)>self.energy_threshold:
            self.num_sample_current_instant=self.num_sample_current_instant+1;#count the number of transmitters
        
        self.is_collision = collision_detection(self.num_sample_current_instant,self.is_collision)
        
        self.iq[-1]=self.iq[-1]+self.channel(loc)*source_iq

    def noise(self):
        return(0+0j); #AWGN to be added

    def channel(self, loc):
        return(1); 

    def stop_receiving_iq(self):
        

        if self.num_sample_current_instant>0:
            self.current_iq_train.append(self.iq[-1])
            if self.frame_ongoing==0:
                print_log("GW", "start of a new frame");
                self.frame_ongoing=1;
        else: #means an idle sample
            print_log("GW", "an Idle sample found");
            if self.frame_ongoing==1:
                print_log("GW", "Tx to Idle transition");
                self.frame_ongoing=0; #get ready for detecting the next start of frame
                self.was_collision=self.is_collision;
                if self.is_collision==0:
                    
                    self.rx=LoRa_Receiver_demodulation(I_Q_sample_physical_layer=self.current_iq_train,SF=self.received_SF)
                    if self.node_num==0:
                        print("received symbol",self.rx, self.received_SF)
                    self.decoded=1;
                else:
                    self.rx=[]; #to ensure that the previous decoded value is not carried over
                self.is_collision=0; #reset so that next frame starts with no collision assumption
                del(self.current_iq_train);
                self.current_iq_train=[];
        del(self.iq)
        self.iq=[];

        if self.was_collision==1: #means an idle sample and also a collision
            self.was_collision=0;
            return("collided");
        if self.decoded==1: #print the message that was received and decoded, when no collision
            self.decoded=0;
            print_log("GW", "decoded: ",self.rx)
            return(self.rx);



def find_thinning_prob(sucess,attempt):
    return(sucess/attempt)


#generate the nodes
nodes=[Node(num=i) for i in range(NumDev)]

gws=[GW() for i in range(NumGW)]

cur_time=int(load_context("cur_time",0));

loc_est=[[] for i in nodes]
    
num_received=[0 for i in nodes];

for j in nodes:
    strn="node"+str(j.num)+"num_received"
    num_received[j.num] = int(load_context(strn,0));


y=0

max_num_events=300000

if cur_time==0:
    for i in nodes:
        print_log("MAIN",",", cur_time,",",i.p,",","not known",",",i.num_attempts,",",i.num)

for i in range(max_num_events): #number of events
    
    time_to_next_event=10000000;
        
    for j in gws:
        j.start_receiving_iq();#new event has happened, add an IQ element to the array at the receiver

    for j in nodes:
        if j.get_last_event_time()+j.get_next_time()<cur_time+time_to_next_event:
            time_to_next_event=j.get_last_event_time()+j.get_next_time()-cur_time;
    cur_time=cur_time+time_to_next_event;
    iq=0;

    for j in nodes:
        if j.get_last_event_time()+j.get_next_time()==cur_time:
            for g in gws: 
                
                g.receive_iq(source_iq=j.get_iq(j.get_samplenum()), loc=j.get_loc(),node=j.get_node_num(),SF=j.get_SF());#new event has happened, add an IQ element to the array at the receiver
            j.do_event();

    for j in gws:
        y=j.stop_receiving_iq();
        if y!=None:#means this was the last IQ sample
            if y!="collided":
                sending_node=y[-1];
                #print_log("MAIN", "One event ended with success",cur_time,sending_node)
                num_received[sending_node]=num_received[sending_node]+1;
                
                if num_received[sending_node]%10 == 0:
                    thinning_probability=(nodes[sending_node].p)*find_thinning_prob(num_received[sending_node],nodes[sending_node].num_attempts);

                    if abs(target_thinning_prob-thinning_probability)>0.05:
                        
                        if thinning_probability<target_thinning_prob:
                            #print_log("MAIN","target is more")
                            nodes[sending_node].p=nodes[sending_node].p+0.1*(target_thinning_prob-thinning_probability) #Increase
                        else:
                            #print_log("MAIN","target is less")
                            nodes[sending_node].p=nodes[sending_node].p+0.1*(target_thinning_prob-thinning_probability) #decrease
                        if nodes[sending_node].p<0.00005:
                            nodes[sending_node].p=0.00005
                        if nodes[sending_node].p>0.9:
                            nodes[sending_node].p=0.9
                    print_log("MAIN",",", cur_time,",",nodes[sending_node].p,",",thinning_probability,",",nodes[sending_node].num_attempts,",",y[-1])
                    # worksheet.write(row, col, nodes[sending_node].p )
                    # worksheet.write(row, col+1, (nodes[sending_node].num_attempts-10)/nodes[sending_node].num_attempts )
                    nodes[sending_node].num_attempts=0
                    num_received[sending_node]=0
                    row=row+1
            #else:
                #print_log("MAIN", "***********************COLLISION*******************************");

    #exit should be at the end when the event before this IDLE event is processed
    if i>int(0.5*max_num_events):
        idle=1
        for j in nodes:
            if j.state!="IDLE":
                idle=0;
        if idle==1:
            #print_log("MAIN","System found to be idle  ", cur_time);
            break;
#workbook.close()
save_context("cur_time",str(cur_time)); #Store the current status of nodes and gateway
#print("cur_time",cur_time)
for j in nodes:
    strn="node"+str(j.num)+"SF"
    save_context(strn,str(j.SF))
    
    strn="node"+str(j.num)+"loc"
    
    save_context(strn,str(j.get_loc()));
    strn="node"+str(j.num)+"p"
    save_context(strn,str(j.p));
    strn="node"+str(j.num)+"next_event"
    save_context(strn,str(j.next_event));
    strn="node"+str(j.num)+"state"
    save_context(strn,str(j.state));
    strn="node"+str(j.num)+"last_event_time"
    save_context(strn,str(j.last_event_time));
    strn="node"+str(j.num)+"num_attempts"
    save_context(strn,str(j.num_attempts));
    strn="node"+str(j.num)+"num_received"
    save_context(strn,str(num_received[j.num]));
    
for g in gws:
    strn="gateway_loc"
    
    save_context(strn, str(g.loc))
