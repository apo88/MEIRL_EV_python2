import numpy as np


def change_dir(policy):
    for pre_s in range(100):
        if(policy[pre_s]==0):
            print pre_s,'r'
        elif(policy[pre_s]==1):
            print pre_s,'l'
        elif(policy[pre_s]==2):
            print pre_s,'d'
        elif(policy[pre_s]==3):
            print pre_s,'u'
        else:
            print pre_s,'s'

def max_dir(policy):
    for pre_s in range(10*10):
        if (np.argmax(policy[pre_s])==0):
            print pre_s, 'r'
        elif (np.argmax(policy[pre_s]==1)):
            print pre_s, 'l'
        elif (np.argmax(policy[pre_s]==2)):
            print pre_s, 'd'
        elif (np.argmax(policy[pre_s]==3)):
            print pre_s, 'u'
        else:
            print pre_s, 's'