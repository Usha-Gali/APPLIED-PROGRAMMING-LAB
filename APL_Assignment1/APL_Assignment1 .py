Python 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
"""     Gali Ushasri
        EE20B033
        EE2703 Applied Programming Lab
            Assignment 1: Solution
"""
import sys
C_START = ".circuit"
C_END = ".end"
def tokens_ext(line):
    allwords = line.split()
     # R, L, C, Independent Sources
    if(len(allwords) == 4):
        elementName = allwords[0]
        from_node = allwords[1]
        to_node = allwords[2]
        value = allwords[3]
        return [elementName, from_node, to_node, value]

    # Current controlled sources
    elif(len(allwords) == 5):
        elementName = allwords[0]
        from_node = allwords[1]
        to_node = allwords[2]
        voltageSource = allwords[3]
        value = allwords[4]
        return [elementName, from_node, to_node, voltageSource, value]

    # Voltage controlled sources
    elif(len(allwords) == 6):
        elementName = allwords[0]
        from_node = allwords[1]
        to_node = allwords[2]
        voltageSourcefrom_node = allwords[3]
        voltageSourceto_node = allwords[4]
        value = allwords[5]
        return [elementName, from_node, to_node, voltageSourcefrom_node, voltageSourceto_node, value]

    else:
        return []

def rev(str): 
    l=str.split()
    l.reverse()
    return " ".join(l)


def print_rev(token_line):
    for x in token_line[::-1]:
        str=x
        print(rev(str))
    print('')
    return

if _name== "_main":
    if len(sys.argv)==2 :# checking for correct file arguement.
        try:
            c_file = sys.argv[1]# accepting file.
            if (not c_file.endswith(".netlist")):#checking if it is netfile.
                print("Please provide correct file type.")
            else:
                with open (c_file, "r") as f:#reading given netfile.
                    lines = []
                    for line in f.readlines():
                        lines.append(line.split('#')[0].split('\n')[0])
                    try:
                        cstart = lines.index(C_START)
                        cend = lines.index(C_END)
                        c_lines = lines[cstart+1:cend]#section that contains circuit.
                        token_line = [tokens_ext(line) for line in c_lines]#getting tokens with help of function.
                        print_rev(token_line)#printing tokens in reverse order.
                    except ValueError:
                        print("please provide correct file.")
        except FileNotFoundError:
            print("please provide correct filename.")
        sys.exit("please provide correct arguments.")
    else:
        sys.exit("please provide correct arguments")
