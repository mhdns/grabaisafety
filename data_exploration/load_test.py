import pandas as pd
import numpy as np
import os
import tensorflow as tf
import warnings 

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

if os.path.isfile('train_data_grouped.h5'):
    train_data = pd.read_hdf('train_data_grouped.h5', 'grabai')
    test_data = pd.read_hdf('val_data_grouped.h5', 'grabai')
else:
    train_data = grouped_data(train_data)
    test_data = grouped_data(test_data)
    train_data.to_hdf('train_data_grouped.h5', 'grabai')
    test_data.to_hdf('val_data_grouped.h5', 'grabai')

train_data = train_data[train_data.label_mean != 0.5]
test_data = test_data[test_data.label_mean != 0.5]


if os.path.isfile('train_data_final.h5'):
    train_data = pd.read_hdf('train_data_final.h5', 'grabai')
    test_data = pd.read_hdf('val_data_final.h5', 'grabai')
else:
    cols = list(train_data)
    cols.remove('label_mean')

    scaler2 = Normalizer()
    scaler2 = scaler2.fit(train_data[cols])
    train_data[cols] = scaler2.transform(train_data[cols])
    test_data[cols] = scaler2.transform(test_data[cols])
    
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()

    train_data.to_hdf('train_data_final.h5', 'grabai')
    test_data.to_hdf('val_data_final.h5', 'grabai')



X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, -1]

pca = PCA(n_components=train_data.shape[1]-2)
pca.fit(X_train)

pca = PCA(n_components=10)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

y_train = np.array(y_train)[:, np.newaxis]
y_test = np.array(y_test)[:, np.newaxis]

# ML Removed


# #### Note
# * New models will be created using the training set and be appended as new data collumns
# * Same will be done for the test set

model1 = LogisticRegression(C=10000).fit(X_train, y_train)
model2 = DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=100).fit(X_train, y_train)
model3 = RandomForestClassifier(criterion='entropy', n_estimators=10).fit(X_train, y_train)
model4 = SVC(probability=True).fit(X_train, y_train)
model5 = GradientBoostingClassifier(subsample=0.7, min_samples_split=10).fit(X_train, y_train)


def get_mod(array, models = [model1, model2, model3, model4, model5]):
    final_model = array
    for i in models:
        final_model = np.concatenate((final_model, i.predict_proba(array)[:, 0:1]), axis=1)
        
    return final_model

new_X_train = get_mod(X_train)
new_X_test = get_mod(X_test)


# #### Note
# * The function below will help to split the safe and unsafe results
# * So that an equal number of safe and unsafe observations can be used in the batches
# * This will improve the training of the neural network a stated in Yann Lecun's paper

def split_data(X, y):
    """Splits data based on classifier 0 and 1"""
    data = np.concatenate((X, y), axis=1)
    zero = data[(data[:, -1]==0)][:, 0:15]
    one = data[(data[:, -1]==1)][:, 0:15]
    
    return (zero, one)

def prep_batch(batch_size, array1, array2):
    holder = []
    zero = np.random.randint(array1.shape[0], size=16)
    one = np.random.randint(array2.shape[0], size=16)
    for i in range(batch_size//2):
        holder.append(array1[zero[i]:zero[i]+1, :])
        holder.append(array2[one[i]:one[i]+1, :])
        
    return np.concatenate(holder, axis=0)



# ### Neural Network

def init_weight(shape):
    weights = tf.truncated_normal(shape, stddev=0.1)
    # Truncated normal will pick values from normal distribution, but if the value
    # is off by 2 std dev, the value is dropped and repicked.
    return tf.Variable(weights)

def init_bias(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)

def layer(input_layer, size):
    # The number of size determines the number of neurons on the next layer
    input_size = int(input_layer.get_shape()[1])
    W = init_weight([input_size, size])
    b = init_bias([size])
  
    return tf.matmul(input_layer, W) + b

# x = tf.placeholder(tf.float32, shape=[None, 15]) 
# y_true = tf.placeholder(tf.float32, shape=[None,1])

# hold_prob = tf.placeholder(tf.float32)
# dropout_layer = tf.nn.dropout(x, rate=1-hold_prob)

# hidden_layer1 = tf.nn.leaky_relu(layer(dropout_layer, 20))

# hidden_layer2 = tf.nn.leaky_relu(layer(hidden_layer1, 20))

# hidden_layer3 = tf.nn.leaky_relu(layer(hidden_layer2, 15))

# hidden_layer4 = tf.nn.leaky_relu(layer(hidden_layer3, 15))

# hidden_layer5 = tf.nn.leaky_relu(layer(hidden_layer4, 10))

# hidden_layer6 = tf.nn.leaky_relu(layer(hidden_layer5, 10))

# hidden_layer7 = tf.nn.leaky_relu(layer(hidden_layer6, 5))

# hidden_layer8 = tf.nn.leaky_relu(layer(hidden_layer7, 5))

# y_hat = tf.nn.softmax(layer(hidden_layer8, 1))

# cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_hat))

# optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
# train = optimizer.minimize(cross_entropy)

# init = tf.global_variables_initializer()

# steps = 1001
# batch_size=32
# saver = tf.train.Saver()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('nn_model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    hold_prob = graph.get_tensor_by_name("hold_prob:0")
    y_hat = graph.get_tensor_by_name("y_hat:0")
    print("Model restored.")

    y_hat = sess.run(y_hat, feed_dict={x: new_X_train, y_true: y_train, hold_prob: 1})

print(classification_report(y_train, np.round(y_hat)))
print(confusion_matrix(y_train, np.round(y_hat)))
