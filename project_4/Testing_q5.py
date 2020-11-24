from NeuralNetUtil import buildExamplesFromCarData, buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt
import csv

def average(argList):
    return sum(argList)/float(len(argList))


def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean), 2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))


penData = buildExamplesFromPenData()


def testPenData(hiddenLayers=[24]):
    return buildNeuralNet(penData, maxItr=200, hiddenLayerList=hiddenLayers)


carData = buildExamplesFromCarData()


def testCarData(hiddenLayers=[16]):
    return buildNeuralNet(carData, maxItr=200, hiddenLayerList=hiddenLayers)


if __name__ == '__main__':
    # Q5:
    with open('Q5results.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Iteration', 'Pen Results', 'Car Results'])
        for i in range(1, 6):
            print("######################################################################################################")
            print("iteration #%d" % i)
            print("######################################################################################################")
            resultPen = testPenData()
            resultCar = testCarData()
            csvwriter.writerow([str(i), str(resultPen[1]), str(resultCar[1])])

