import csv
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, \
    confusion_matrix
from sklearn import preprocessing

pt_split_sb = np.vectorize(lambda t: t.split('[')[0])
pt_split_rb = np.vectorize(lambda t: t.split('(')[0])


class PosTagEval:
    def __init__(self, gold_file, pred_file):
        self.gold_file = gold_file
        self.pred_file = pred_file
        self.gold_tags = []
        self.pred_tags = []
        self.labels = []

    def get_tags(self):
        if self.gold_file == '' or self.pred_file == '':
            print('Please provide file names.')
            return

        le = preprocessing.LabelEncoder()

        # orig file
        df = pd.read_csv(self.gold_file, delimiter='\t', header=None,
                         engine='python', quoting=csv.QUOTE_NONE)
        pt = df[0].to_numpy()
        pt = pt[pt != None]
        # pt = pt_split_sb(pt)
        le.fit(pt)
        self.labels = list(le.classes_)
        self.gold_tags = pt
        # print(self.labels)

        # new file
        df = pd.read_csv(self.pred_file, delimiter='\t',
                         skiprows=1, header=None,
                         engine='python', quoting=csv.QUOTE_NONE)
        # print(df)
        pt = df[4].to_numpy()
        pt = pt[pt != None]

        if '(' in pt[0]:
            self.pred_tags = pt_split_rb(pt)
        elif '[' in pt[0]:
            self.pred_tags = pt_split_sb(pt)
        else:
            self.pred_tags = pt

    def report(self):
        print(classification_report(self.gold_tags, self.pred_tags, labels=self.labels))
        cm = confusion_matrix(self.gold_tags, self.pred_tags, labels=self.labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)
        disp.plot()
        plt.show()


try:
    _, gold_file, pred_file = sys.argv
except ValueError:
    print('Usage: python3 pos_eval.py /path/to/gold/file /path/to/pred/file')

pte = PosTagEval(gold_file, pred_file)
pte.get_tags()
pte.report()
