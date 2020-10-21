"""
Created on Feb 21, 2011
Machine Learning in Action Chapter 18
Map Reduce Job for Hadoop Streaming 
mrMeanMapper.py
@author: Peter Harrington

cd ch15
python mrMeanMapper.py < inputFile.txt
"""
import sys
from numpy import mat, mean, power


def read_input(file):
    for line in file:
        yield line.rstrip()


# creates a list of input lines
my_input = read_input(sys.stdin)
my_input = [float(line) for line in my_input]    # overwrite with floats
numInputs = len(my_input)
my_input = mat(my_input)
sqInput = power(my_input, 2)
# output size, mean, mean(square values)
print("%d\t%f\t%f" % (numInputs, mean(my_input), mean(sqInput)))   # calc mean of columns
print("report: still alive", file=sys.stderr)
