
import numpy as np
from sys import argv, exit


circuit_start = ".circuit"
circuit_end = ".end"
AC = '.ac'

class element_def(): #defining class 
    def __init__(self,line):
        self.line = line
        self.tokens = self.line.split()
        self.name = typeof_element(self.tokens[0])
        self.from_node = self.tokens[1]
        self.to_node = self.tokens[2]

        if len(self.tokens) == 6:
            self.type = 'AC'
            Vm = np.double(self.tokens[4])/2
            phase = np.double(self.tokens[5])
            real = Vm * np.cos(phase)
            imag = Vm * np.sin(phase)
            self.value = complex(real,imag)
        elif len(self.tokens) == 5:
            self.type = 'DC'
            self.value = np.double(self.tokens[4])

        else:
            self.type = 'DC-only'
            self.value = np.double(self.tokens[3])


def typeof_element(token): #getting name of the element
    element_fromCODE = {"R": "resistor", "L": "inductor", "C": "capacitor", "V": "ind voltage source", "I": "ind current source" }
    return element_fromCODE.get(token[0], None)


def frequency(lines):#calculating frequency 
    k = 0
    for line in lines:
        if line[:3] == '.ac':
            k = np.double(line.split()[2])
    return k

def getting_key(dict,value):#getting corresponding keys for particular values
    for key in dict.keys():
        if dict[key] == value :
            return key

def mapping_nodes(circuit): #finding and returning nodes dictionary 
    dict = {"GND" : 0} 
    nodes = [element_def(line).from_node for line in circuit]
    nodes += [element_def(line).to_node for line in circuit]
    nodes = list(set(nodes)) 

    #printing(nodes)
    cnt = 1
    for node in nodes:
        if node != 'GND' :
            dict[node] = cnt
            cnt += 1
    return dict

def building_dictionary(circuit,e):#making dictionary for all components
    dict = {}
    element_names = [element_def(line).tokens[0] for line in circuit if element_def(line).tokens[0][0].lower()== e]
    for i, name in enumerate(element_names):
        dict[name] = i
    return dict


def finding_nodes(circuit, node_key, node_map):#finding lines 
    INDs = []
    for i in range(len(circuit)):
        for j in range(len(circuit[i].split())):
            if circuit[i].split()[j] in node_map.keys():
                if node_map[circuit[i].split()[j]] == node_key:
                    INDs.append((i, j))

    return INDs


def matrix_update(node_key): #updating matrix for given nodes
    INDs = finding_nodes(circuit, node_key, node_map)
    for ind in INDs:
       #getting all elements attributes
        element = element_def(circuit[ind[0]])
        name_of_element = circuit[ind[0]].split()[0]
       #INDEPENDENT VOLTAGE SOURCE
        if name_of_element[0] == 'V' :
            index = volt_d[name_of_element]
            if ind[1]== 1:
                neig_key = node_map[element.to_node]
                C[node_key,n+index] += 1
                C[n+index,node_key] -= 1
                D[n+index] = element.value
            if ind[1] == 2 :
                neig_key = node_map[element.from_node]
                C[node_key,n+index] -= 1
                C[n+index,node_key] +=1
                D[n+index] = element.value
       #INDEPENDENT CURRENT SOURCE
        if name_of_element[0] == 'I' :
            if ind[1]== 1:
                D[node_key] -= element.value
            if ind[1] == 2 :
                D[node_key] += element.value
        #CAPACITORS
        if name_of_element[0] == 'C' :
            if ind[1]== 1: 
                neig_key = node_map[element.to_node]
                C[node_key, node_key] += complex(0, 2 * np.pi * k * (element.value))
                C[node_key, neig_key] -= complex(0, 2 * np.pi * k * (element.value))
            if ind[1] == 2 :
                neig_key = node_map[element.from_node]
                C[node_key, node_key] += complex(0, 2 * np.pi * k * (element.value))
                C[node_key, neig_key] -= complex(0, 2 * np.pi * k * (element.value))
           
        
        #RESISTORS
        if name_of_element[0] == 'R':
            if ind[1] == 1: 
                neig_key = node_map[element.to_node]
                C[node_key, node_key] += 1/(element.value)
                C[node_key, neig_key] -= 1/(element.value)
                    
            if ind[1] == 2 : 
                neig_key = node_map[element.from_node]
                C[node_key, node_key] += 1/(element.value)
                C[node_key, neig_key] -= 1/(element.value)
        #INDUCTORS   
        if name_of_element[0] == 'L' :
            try:
                if ind[1]== 1:
                    neig_key = node_map[element.to_node]
                    C[node_key, node_key] -= complex(0,1/(2 * np.pi * k * element.value))
                    C[node_key, neig_key] += complex(0,1/(2 * np.pi * k * element.value))
                if ind[1] == 2 :
                    neig_key = node_map[element.from_node]
                    C[node_key, node_key] -= complex(0,1/(2 * np.pi * k * element.value))
                    C[node_key, neig_key] += complex(0,1/(2 * np.pi * k * element.value))
            except ZeroDivisionError: 
                idx = ind_d[name_of_element]
                if ind[1]== 1:
                    neig_key = node_map[element.to_node]
                    C[node_key, n + k + idx] += 1 
                    C[n + k + idx, node_key] -= 1
                    D[n + k + idx] = 0
                if ind[1]== 2:
                    C[node_key, n + k + idx] -= 1
                    C[n + k + idx, node_key] += 1
                    D[n + k + idx] = 0
    
#start of the main function
if len(argv) != 2:
    print('\nUsage: %s <inputfile>' % argv[0]) #acceptinmg netlist file name and checking if user has access
    exit()
try:
    with open(argv[1]) as f:
        lines = f.readlines()
        k = frequency(lines) #getting freq of siyrce 
        start = -1
        end = -2
        for line in lines:   #extracting circuit definition          
            if circuit_start == line[:len(circuit_start)]:
                start = lines.index(line)
            elif circuit_end == line[:len(circuit_end)]:
                end = lines.index(line)
                break
        if start >= end:           
            print('circuit definition is invalid')
            exit(0)

        
        circuit = []
        for line in [' '.join(line.split('#')[0].split()) for line in lines[start+1:end]]:
            circuit.append(line)                
        

        node_map = mapping_nodes(circuit)
       

        volt_d = building_dictionary(circuit, "v")
        ind_d = building_dictionary(circuit,'l')
        
        k = len([i for i in range(len(circuit)) if circuit[i].split()[0][0] == 'V'])
        n = len(node_map)
        dim = n + k   #checking if source is AC or DC 
        
        if k == 0: #
            C = np.zeros((dim+len(ind_d),dim+len(ind_d)),dtype=np.complex)
            D = np.zeros(dim+len(ind_d),dtype=np.complex)
        else:
            C = np.zeros((dim,dim),dtype=np.complex)
            D = np.zeros(dim,dtype=np.complex)

        for i in range(len(node_map)): 
            matrix_update(i)
        C[0] = 0
        C[0,0] =1

        #constructing C and D arrays
        print('The node dictionary is :',node_map)
        print('C = :\n',C)
        print('D = :\n',D)


       #solving Cx=D
        try:
            x = np.linalg.solve(C,D)    
        except Exception:
            print('The incidence matrix cannot be inverted as it is singular.')
            sys.exit()

        print('Voltage convention -> From node is at a lower potential')     
        
        for i in range(n):
            print("Voltage at NODE {} is {}".format(getting_key(node_map,i),x[i]))
        for j in range(k):
            print('Current through SOURCE {} is {}'.format(getting_key(volt_d,j),x[n+j]))
        if k == 0:
            for i in range(len(ind_d)):
                print("Current through INDUCTOR {} is {}".format(getting_key(ind_d,i),x[n+k+i]))

#checking if there is any error
except IOError:
    print('file is invalid')
    exit()
