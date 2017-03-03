import csv
import numpy

filename ='pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rb')

# The example loads an object that can iterate over each row of the data
# and can easily be converted into a NumPy array.
# Running the example prints the shape of the array: (768, 9) (rows, columns)

#reader = csv.reader(raw_data, delimiter = ',', quoting = csv.QUOTE_NONE)
#x = list(reader)
#data = numpy.array(x).astype('float')
#print (data.shape)

#load file with numpy
#Running the example will load the file as a numpy.ndarray
# and print the shape of the data (768,9) the same

data = numpy.loadtxt(raw_data, delimiter=",")

print (data.shape)

# Load CSV from URL using NumPy

import urllib
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data = urllib.urlopen(url)
dataset = numpy.loadtxt(raw_data, delimiter=",")
print(dataset.shape)

# load csv with Pandas

import pandas
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass' ,'pedi', 'age', 'class']
datapandas = pandas.read_csv(filename, names=names)

print (data.shape)

#or pandas directly from url

urlpandas = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(urlpandas, names=names)
print(data.shape)
