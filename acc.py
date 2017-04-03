import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="./prediction.csv")
args = parser.parse_args()


ans = open('./data/test_answer.csv', 'r')
ans_list = []
for i, row in enumerate(csv.reader(ans)):
	if i == 0:
		continue
	else:
		ans_list.append(row[1])

prd = open(args.input, 'r')
prd_list = []
for i, row in enumerate(csv.reader(prd)):
	if i == 0:
		continue
	else:
		prd_list.append(row[1])

acc = 0.
for i in range(len(prd_list)):
	if prd_list[i] == ans_list[i]:
		acc += 1.

print ('Accuracy: {}'.format(acc/float(len(prd_list))))
