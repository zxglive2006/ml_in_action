"""
Created on Feb 21, 2011
@author: Peter

cd ch15
python mrMeanMapper.py < inputFile.txt | python mrMeanReducer.py
"""
import sys


def read_input(file):
    for line in file:
        yield line.rstrip()


# creates a list of input lines
my_input = read_input(sys.stdin)

# split input lines into separate items and store in list of lists
mapperOut = [line.split('\t') for line in my_input]

# accumulate total number of samples, overall sum and overall sum sq
cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj * float(instance[1])
    cumSumSq += nj * float(instance[2])

# calculate means
mean = cumVal / cumN
meanSq = cumSumSq / cumN

# output size, mean, mean(square values)
print("%d\t%f\t%f" % (cumN, mean, meanSq))
print("report: still alive", file=sys.stderr)
