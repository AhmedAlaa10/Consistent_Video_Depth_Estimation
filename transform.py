import os
import sys
dataset="sintel" #"sintel"/"RGBD"
if len(sys.argv) > 1:
    dataset = str(sys.argv[1])
command = 'python tranform_'+dataset+'.py'
i=2
while(len(sys.argv) > i):
    command+=' '+ sys.argv[i]
    i+=1
os.system(command)