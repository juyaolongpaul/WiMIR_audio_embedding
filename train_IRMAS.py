from sklearn.svm import SVC
import os
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


feature_x = []
feature_y = []
counter = 0
directory = ['irmas-vggish', 'irmas-openl3']


def train_using_embedding(folder_name, counter, feature_x, feature_y):
    print('Step 1: reading files')
    for id, file in enumerate(os.listdir(os.path.join('.', 'Data', folder_name))):
        print(id, file)
        ptr = file.find('[')
        ground_truth = file[ptr + 1: ptr + 4]
        with open(os.path.join('.', 'Data', folder_name, file)) as json_file:  # put features in a 2D matrix
            data = json.load(json_file)
            lists = data['features']
            for id, row in enumerate(lists):
                if counter == 0:
                    feature_x = np.concatenate((feature_x, row))
                    feature_y = np.concatenate((feature_y, [ground_truth]))
                    counter += 1
                else:
                    feature_x = np.vstack((feature_x, row))
                    feature_y = np.vstack((feature_y, [ground_truth]))
    print('Step 2: process data for ML models')
    pd.DataFrame(data=feature_y)
    le = preprocessing.LabelEncoder()
    feature_y_panda = pd.DataFrame(data=feature_y)
    feature_yy = feature_y_panda.apply(le.fit_transform)
    print(feature_y_panda)
    print(feature_yy)
    enc = preprocessing.OneHotEncoder()
    enc.fit(feature_yy)
    model = SVC(verbose=True, kernel='poly')
    x_train, x_test, y_train, y_test = train_test_split(feature_x, feature_yy, test_size=0.2, random_state=0)
    print('Step 3: training')
    model.fit(x_train, y_train.to_numpy().ravel())
    y_predict = model.predict(x_test)
    print('Step 4: predicting')
    test_acc = accuracy_score(y_test, y_predict)
    print(test_acc)


if __name__ == "__main__":
    train_using_embedding(directory[0], counter, feature_x, feature_y)


