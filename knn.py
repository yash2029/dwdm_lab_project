import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import itertools

class K_Nearest_Neighbors_Classifier() :
    def __init__( self, K ) :
        self.K = K
    
    def fit( self, X_train, Y_train ) :
        self.X_train = X_train
        self.Y_train = Y_train
        self.m, self.n = X_train.shape

    def predict( self, X_test ) :
        self.X_test = X_test
        self.m_test, self.n = X_test.shape
        Y_predict = np.zeros( self.m_test )
        print(self.m_test)
        for i in range( self.m_test ) :
            x = self.X_test[i]
            neighbors = np.zeros( self.K )
            neighbors = self.find_neighbors( x )
            print('Prediction'+str(i)+": " , end=' ')
            Y_predict[i] = mode( neighbors )[0][0]
            print(str(Y_predict[i]))
        return Y_predict

         
    def find_neighbors( self, x ) :
        euclidean_distances = np.zeros( self.m )
        for i in range( self.m ) :
            d = self.euclidean( x, self.X_train[i] )
            euclidean_distances[i] = d
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[inds]
        return Y_train_sorted[:self.K]

 
    def euclidean( self, x, x_train ) :
        return np.sqrt( np.sum( np.square( x - x_train ) ) )



def status_to_int(s):
	if (s == 'benign'): return 0
	if (s == 'malware'): return 1

# combine all benign and malware csv files csv files
os.system('find ./CSV_FILES_BATCH/ -name "*_benign.csv" > ./CSV_FILES_PATH_LIST/benign_batch_csv_files_list.txt')
os.system('find ./CSV_FILES_BATCH/ -name "*_malware.csv" > ./CSV_FILES_PATH_LIST/malware_batch_csv_files_list.txt')

benign_file_list = open('./CSV_FILES_PATH_LIST/benign_batch_csv_files_list.txt').read()
malware_file_list = open('./CSV_FILES_PATH_LIST/malware_batch_csv_files_list.txt').read()

benign_file_list = benign_file_list.split('\n')[0:-1]
malware_file_list = malware_file_list.split('\n')[0:-1]

# combine all files list
benign_csv_combined = pd.concat([pd.read_csv(f) for f in benign_file_list ], ignore_index = True)
malware_csv_combined = pd.concat([pd.read_csv(f) for f in malware_file_list ], ignore_index = True)

# export to csv
benign_csv_combined.to_csv( "./CSV_FILES_BATCH/benign.csv", index = False)
malware_csv_combined.to_csv( "./CSV_FILES_BATCH/malware.csv", index = False)

# read combined dataframes
df_benign = pd.read_csv('./CSV_FILES_BATCH/benign.csv')
df_malware = pd.read_csv('./CSV_FILES_BATCH/malware.csv')

df = [df_benign, df_malware]
df = pd.concat(df, ignore_index = True)

df = df.fillna(0)

df['status'] = df['status'].apply(status_to_int)

X_drop_columns = ['status']

drop = ['ip.src.len.entropy', 'ip.src.len.cvq','ip.dst.len.entropy', 'ip.dst.len.cvq', 'sport.entropy', 'sport.cvq', 'dport.entropy', 'dport.cvq', 'tcp.flags.entropy', 'tcp.flags.cvq',]

X_drop_columns_more = ['ip.src.len.median', 'ip.src.len.var', 'ip.src.len.std','ip.src.len.cv', 'ip.src.len.rte', 'ip.dst.len.median', 'ip.dst.len.var', 'ip.dst.len.std', 'ip.dst.len.cv', 'ip.dst.len.rte','tcp.flags.mean', 'tcp.flags.median', 'tcp.flags.var', 'tcp.flags.std', 'tcp.flags.entropy', 'tcp.flags.cv', 'tcp.flags.cvq', 'tcp.flags.rte']

y_drop_columns = ['ip.proto', 'ip.src.len.mean', 'ip.src.len.median', 'ip.src.len.var', 'ip.src.len.std', 'ip.src.len.entropy', 'ip.src.len.cv', 'ip.src.len.cvq', 'ip.src.len.rte', 'ip.dst.len.mean', 'ip.dst.len.median', 'ip.dst.len.var', 'ip.dst.len.std', 'ip.dst.len.entropy', 'ip.dst.len.cv', 'ip.dst.len.cvq', 'ip.dst.len.rte', 'sport.mean', 'sport.median', 'sport.var', 'sport.std', 'sport.entropy', 'sport.cv', 'sport.cvq', 'sport.rte', 'dport.mean', 'dport.median', 'dport.var', 'dport.std', 'dport.entropy', 'dport.cv', 'dport.cvq', 'dport.rte', 'tcp.flags.mean', 'tcp.flags.median', 'tcp.flags.var', 'tcp.flags.std', 'tcp.flags.entropy', 'tcp.flags.cv', 'tcp.flags.cvq', 'tcp.flags.rte']

X = df.drop(drop + X_drop_columns_more + ['status'], axis = 1).values
y = df.drop(y_drop_columns, axis = 1).values

# convert dtype to int
X = X.astype(float)
y = y.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/1000, random_state = 0)

#  == PREPROCESSING ==
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# == TRAINING ==

knn_model = K_Nearest_Neighbors_Classifier(3)
knn_model.fit(X_train_scaled, y_train.ravel())
filename = 'batch_network_traffic_knn_classifier.sav'
pickle.dump(knn_model , open(filename, 'wb'))
print('classifying...')
y_pred = knn_model.predict(X_test_scaled)
correctly_classified = 0
count = 0
tp = tn = fp = fn = 0
for i in range(np.size(y_pred)):
    if y_test[i] == 1:
        if y_pred[i] == 1:
            correctly_classified+=1
            tp+=1
        else:
            fn+=1
    else:
        if y_pred[i] == 0:
            correctly_classified+=1
            tn+=1
        else:
            fp+=1
    count+=1
confusion_matrix = [[tp,fp],[fn,tn]] 
# print accuracy
print("Accuracy: " + str((correctly_classified/count)*100))



# == PRINT METRICS ==

# Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.asarray(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print('Confusion matrix')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Compute confusion matrix
np.set_printoptions(precision=2)

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(confusion_matrix, classes=['Safe Trafic', 'DDoS Traffic'], title='Confusion matrix, without normalization')
