# Lewis Shemery 1001402131

import math
import sys

class naive_bayes:
    def __init__(self, train_file, test_file):
        self.classifier_dic = {}
        self.mean_dic = {}
        self.stdev_dic = {}
        self.normal_dic = []
        self.unique_labels = []
        self.probability_dic = {}
        self.train_file = train_file
        self.test_file = test_file

    def train(self):
        train_file_path = self.train_file
        data_dic, self.unique_labels, total_number_of_rows, data_list = load_training_data(train_file_path)
        self.unique_labels = list(map(int, self.unique_labels))
        self.unique_labels = sorted(self.unique_labels)
        for label in self.unique_labels:
            for dimension in range(0, len(data_dic[label][0])):
                gaussian_training = Training(data_dic[label], dimension)
                mean = gaussian_training.get_mean()
                stdev = gaussian_training.get_stdev()
                if stdev < 0.0001:
                    stdev = 0.0001
                if label in self.mean_dic:
                    self.mean_dic[label].append(mean)
                    self.stdev_dic[label].append(stdev)
                else:
                    self.mean_dic[label] = [mean]
                    self.stdev_dic[label] = [stdev]
                print("Class %d, dimension %d, mean = %.2f, std = %.2f" % (
                    label, dimension + 1, mean, math.sqrt(stdev)))
        self.probability_dic = Training.probability_of_classifiers(self.unique_labels, total_number_of_rows, data_dic)

    def test(self):
        test_file_path = self.test_file
        number_of_test_rows, test_data = load_test_data(test_file_path)
        classification = Testing(self.unique_labels, self.mean_dic, self.stdev_dic, self.probability_dic)
        for row in test_data:
            classification.classify(row)

        classification.display_accuracy(number_of_test_rows)

class Training:
    def __init__(self, data, y):
        self.data_list = data
        self.column_number = y
        self.mean = 0
        self.sigma = 0
        self.stdev = 0

    def get_mean(self):
        summation = 0
        for elements in self.data_list:
            summation += elements[self.column_number]
        self.mean = summation / len(self.data_list)
        return self.mean

    def get_stdev(self):
        if len(self.data_list) == 0:
            return 0
        summation = 0
        for elements in self.data_list:
            summation += ((elements[self.column_number] - self.mean) * (elements[self.column_number] - self.mean))
        self.sigma = math.sqrt(summation / (len(self.data_list) - 1))
        self.stdev = math.pow(self.sigma, 2)
        return self.stdev

    def get_normal_distribution(self, label):
        if self.stdev == 0:
            return 0
        denominator = self.sigma * (math.sqrt(2 * math.pi))
        power = ((-1) * math.pow((label - self.mean), 2)) / (2 * self.stdev)
        numerator = math.pow(math.e, power)
        normal = numerator / denominator
        return normal

    def probability_of_classifiers(unique_labels, total_number_of_rows, data_dic):
        probability_dic = {}
        for labels in unique_labels:
            probability = len(data_dic[labels]) / float(total_number_of_rows)
            probability_dic[labels] = probability
        return probability_dic

class Testing:
    def __init__(self, unique_labels, mean, stdev, probability_dic):
        self.unique_labels = unique_labels
        self.mean_dic = mean
        self.stdev_dic = stdev
        self.normal_dic = {}
        self.probability_dic = probability_dic
        self.correctly_classifed = 0
        self.total_probability = {}

    def classify(self, data_row):
        self.normal_dic = {}
        for label in self.unique_labels:
            for column in range(0, len(data_row) - 2):
                normal_result = self.get_normal_distribution(data_row[column], self.mean_dic[label][column], self.stdev_dic[label][column])
                if label in self.normal_dic:
                    self.normal_dic[label] *= normal_result
                else:
                    self.normal_dic[label] = normal_result
        maximum = -1
        for label in self.normal_dic:
            self.normal_dic[label] = (self.normal_dic[label] * self.probability_dic[label])
        denominator = sum(self.normal_dic.values())
        for label in self.normal_dic:
            self.normal_dic[label] /= (float(denominator))
            if maximum < self.normal_dic[label]:
                maximum = self.normal_dic[label]
                classified_as = label

        accuracy = 0
        if classified_as == data_row[-2]:
            self.correctly_classifed += 1
            accuracy = 1
        print("ID = %5d, predicted = %3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (
            data_row[-1], classified_as, maximum, data_row[-2], accuracy))

    def display_accuracy(self, number_of_test_rows):
        print("classification accuracy=%6.4lf " % (self.correctly_classifed / float(number_of_test_rows)))

    @staticmethod
    def get_normal_distribution(value, mean, stdev):
        if stdev < 0.01:
            stdev = 0.01
        denominator = math.sqrt(stdev * 2 * math.pi)

        power = ((-1) * math.pow((value - mean), 2)) / float((2 * stdev))
        numerator = math.pow(math.e, power)
        normal = numerator / float(denominator)

        return normal

def load_training_data(filename):
    data_list = []
    input_file = open(filename, "r")
    unique_labels = []
    dictionary = {}
    count = 0
    for line in input_file:
        row_list = (line.split(" "))
        row_list = list(filter(None, row_list))
        row_list = list(map(float, row_list))
        data_list.append(row_list)
        if row_list[-1] in dictionary:
            dictionary[row_list[-1]].append(row_list[0:-1])
        else:
            unique_labels.append(row_list[-1])
            dictionary[row_list[-1]] = [row_list[0:-1]]
        count += 1
    return dictionary, unique_labels, count, data_list

def load_test_data(filename):
    data_list = []
    input_file = open(filename, "r")
    count = 0
    for line in input_file:
        row_list = (line.split(" "))
        row_list = list(filter(None, row_list))
        row_list = list(map(float, row_list))
        data_list.append(row_list + [count])
        count += 1
    return count, data_list


def main():
    training_file = sys.argv[1]
    test_file = sys.argv[2]

    gaussian = naive_bayes(training_file, test_file)
    gaussian.train()
    gaussian.test()

main()