import pandas as pd
import numpy as np
import os
import tensorflow as tf
import warnings 
import pickle
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')

print(tf.__version__)

curr_dir = os.getcwd()
print(curr_dir)

train_data = pd.read_hdf('train_data.h5', 'grabai')
train_data.Speed = train_data.Speed.replace(-1,np.NaN)
test_data = pd.read_hdf('val_data.h5', 'grabai')
test_data.Speed = test_data.Speed.replace(-1,np.NaN)

# Simple fillna would do as only 1.5% of data is missing
train_data.Speed.fillna(train_data.Speed.median(), inplace=True)
test_data.Speed.fillna(test_data.Speed.median(), inplace=True)

# ### Notes
# <div style="width: 400px; float: right;">![image.png](attachment:image.png)</div>
# #### Acceleration
# * Acceleration X shows acceleration on the left or right (if value more than 0, it is a left turn)
# * Accelleration Y is shows acceleration on the up down axis (due to gravity, default will be -9.81, therefore if value more than -9.81, then the vehicle/phone is down)
# * Axeleration Z is the forward and backward acceleration (if less than 0, it is foreward acceleration. This will be the case most of the time)
# 
# #### Gyro
# * Gyro X rotation along the X axis (pitch). Positive value refers to a left tilt on the X axis.
# * Gyro Y rotation along the Y axis (roll). Positive value refers to a left tilt on the Y axis.
# * Gyro Z rotation along the Z axis (yaw). Positive value refers to a left tilt on the Z axis.
# 
# #### Interpretation
# * A left/right turn will have change in acceleration along the X axis, while changing the Roll(gyro Y axis). The larger the gyro Y value the sharper the turn.
# * A up/down movement will have change in acceleration along the Y axis, while changing the Pitch(gyro X axis). The larger the gyro X value the steeper the slope.
# * When there is a combination of left/right and up/down or the road is tilted, the Yaw(gyro Z axis) will change. The value represents shows if it is a right(positive) or left(negative) tilt.

def add_features(df):
    # this function adds the direction, smoothness and intensity for each data point
    # Direction
    df['up'] = df.acceleration_y.apply(lambda y: 1 if np.round(y,2)+9.81 < 0 else 0)
    df['down'] = df.acceleration_y.apply(lambda y: 1 if np.round(y,2)+9.81 > 0 else 0)
    
    df['right'] = df.acceleration_x.apply(lambda x: 1 if np.round(x,2) < 0 else 0)
    df['left'] = df.acceleration_x.apply(lambda x: 1 if np.round(x,2) > 0 else 0)
    
    # Though the else value for smoothness and intensity is not technically correct, 
    # it should help to create some noise that will improve generalization
    # Smoothness
    df['rl_smooth'] = np.where((df.Speed!=0) & (df.gyro_y!=0),
                               df.Speed/np.abs(df.gyro_y),
                               df.Speed + np.abs(df.gyro_y)) # right left smoothness
    
    df['ud_smooth'] = np.where((df.Speed!=0) & (df.gyro_x!=0),
                               df.Speed/np.abs(df.gyro_x),
                               df.Speed + np.abs(df.gyro_x)) # up down smoothness
    
    df['smoothness'] = np.sqrt(np.square(df.rl_smooth) + np.square(df.ud_smooth))
    
    # Intensity
    df['rl_intensity'] = np.where(df.acceleration_x != 0, 
                                  df.rl_smooth * np.abs(df.acceleration_x), 
                                  df.rl_smooth)
    df['ud_intensity'] = np.where(df.acceleration_x != 0, 
                                  df.ud_smooth * np.abs(df.acceleration_y), 
                                  df.ud_smooth)
    df['intensity'] = np.sqrt(np.square(df.rl_intensity) + np.square(df.ud_intensity))
    
    df_col = list(df)
    df_col.remove('label')
    df_col.append('label')
    
    return df[df_col]

if os.path.isfile('train_data_features.h5'):
    train_data = pd.read_hdf('train_data_features.h5', 'grabai')
    test_data = pd.read_hdf('val_data_features.h5', 'grabai')
else:
    train_data = add_features(train_data)
    test_data = add_features(test_data)
    
    train_data.to_hdf('train_data_features.h5', 'grabai')
    test_data.to_hdf('val_data_features.h5', 'grabai')


from sklearn.preprocessing import Normalizer

if os.path.isfile('train_data_features.h5'):
    train_data = pd.read_hdf('train_data_norm.h5', 'grabai')
    test_data = pd.read_hdf('val_data_norm.h5', 'grabai')
else:
    cols = list(train_data)
    cols.remove('label')
    cols.remove('bookingID')

    scaler = Normalizer()
    scaler = scaler.fit(train_data[cols])
    train_data[cols] = scaler.transform(train_data[cols])
    test_data[cols] = scaler.transform(test_data[cols])
    
    train_data.to_hdf('train_data_norm.h5', 'grabai')
    test_data.to_hdf('val_data_norm.h5', 'grabai')

def grouped_data(df):
    extra_features = []
    all_cols = list(df)
    cols = all_cols
    cols.remove('bookingID')
    cols.remove('label')
    
    grouped = df.groupby('bookingID')
    lst_functions = ['mean', 'median', 'min', 'max', 'std', 'skew', 'count', 'sum']
    for func in lst_functions:
        if func == 'mean':
            temp = grouped.mean()
            temp.columns = list(map(lambda x: x + '_mean', list(temp)))
            extra_features.append(temp)
            
        if func == 'median':
            temp = grouped.median()
            temp = temp[cols]
            temp.columns = list(map(lambda x: x + '_median', list(temp)))
            extra_features.append(temp)
        
        if func == 'std':
            temp = grouped.std()
            temp = temp[cols]
            temp.columns = list(map(lambda x: x + '_std', list(temp)))
            extra_features.append(temp)
        
        if func == 'skew':
            temp = grouped.skew()
            temp = temp[cols]
            temp.columns = list(map(lambda x: x + '_skew', list(temp)))
            extra_features.append(temp)
        
        if func == 'count':
            temp = grouped.count()
            temp = temp[cols]
            temp.columns = list(map(lambda x: x + '_count', list(temp)))
            extra_features.append(temp)
        
        if func == 'sum':
            temp = grouped.sum()
            temp = temp[cols]
            temp.columns = list(map(lambda x: x + '_sum', list(temp)))
            extra_features.append(temp)
        
        if func == 'min':
            temp = grouped.min()
            temp = temp[cols]
            temp.columns = list(map(lambda x: x + '_min', list(temp)))
            extra_features.append(temp)
            
        if func == 'max':
            temp = grouped.max()
            temp = temp[cols]
            temp.columns = list(map(lambda x: x + '_max', list(temp)))
            extra_features.append(temp)
            
    merged_data = pd.concat(extra_features, axis=1)
        
    df_col = list(merged_data)
    df_col.remove('label_mean')
    df_col.append('label_mean')
    
    return merged_data[df_col]


if os.path.isfile('train_data_grouped.h5'):
    train_data = pd.read_hdf('train_data_grouped.h5', 'grabai')
    test_data = pd.read_hdf('val_data_grouped.h5', 'grabai')
else:
    train_data = grouped_data(train_data)
    test_data = grouped_data(test_data)
    train_data.to_hdf('train_data_grouped.h5', 'grabai')
    test_data.to_hdf('val_data_grouped.h5', 'grabai')

# #### Note
# * It appears that there are bookingID where label is both one and zero (safe and unsafe)
# * Since these observations are a small portion of the whole data set they will be removed


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

all_data = pd.read_hdf('data_final.h5', 'grabai')

from sklearn.decomposition import PCA

X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]

X = all_data.iloc[:, 1:-1]
y = all_data.iloc[:, -1]

X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, -1]

pca = PCA(n_components=train_data.shape[1]-2)
pca.fit(X_train)

pca = joblib.load('pca_fit.sav')

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

y_train = np.array(y_train)[:, np.newaxis]
y_test = np.array(y_test)[:, np.newaxis]

X = pca.transform(X)
y = np.array(y)[:, np.newaxis]

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


# Since 10000 gave the best f1 score, set C=10000
lr = LogisticRegression(C=10000)
lr.fit(X_train, y_train)
y_hat_lr = lr.predict(X_test)
print("LogisticRegression")
print(classification_report(y_test, y_hat_lr))
print(confusion_matrix(y_test, y_hat_lr))

# ### Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifier = DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=100)
classifier.fit(X_train, y_train)
y_hat_tree = classifier.predict(X_test)
print("DecisionTreeClassifier")
print(classification_report(y_test, y_hat_tree))
print(confusion_matrix(y_test, y_hat_tree))



classifier = RandomForestClassifier(criterion='entropy', n_estimators=10)
classifier.fit(X_train, y_train)
y_hat_rf = classifier.predict(X_test)
print("RandomForestClassifier")
print(classification_report(y_test, y_hat_rf))
print(confusion_matrix(y_test, y_hat_rf))

from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)
y_hat_svm = classifier.predict(X_test)
print("SVC")
print(classification_report(y_test, y_hat_svm))
print(confusion_matrix(y_test, y_hat_svm))

from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(subsample=0.7, min_samples_split=10)
classifier.fit(X_train, y_train)
y_hat_grad = classifier.predict(X_test)
print("GradientBoostingClassifier")
print(classification_report(y_test, y_hat_grad))
print(confusion_matrix(y_test, y_hat_grad))


# #### Note
# * New models will be created using the training set and be appended as new data collumns
# * Same will be done for the test set

# model1 = joblib.load('logistic.sav')
# model2 = joblib.load('decision_tree.sav')
# model3 = joblib.load('random_forest.sav')
# model4 = joblib.load('svc.sav')
# model5 = joblib.load('knn.sav')

model1 = LogisticRegression(C=10000).fit(X, y)
model2 = DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=100).fit(X, y)
model3 = RandomForestClassifier(criterion='entropy', n_estimators=10).fit(X, y)
model4 = SVC(probability=True).fit(X, y)
model5 = KNeighborsClassifier(n_neighbors=15).fit(X, y)


def get_mod(array, models = [model1, model2, model3, model4, model5]):
    final_model = array
    for i in models:
        final_model = np.concatenate((final_model, i.predict_proba(array)[:, 0:1]), axis=1)
        
    return final_model
new_X_train = get_mod(X_train)
new_X_test = get_mod(X_test)
X = get_mod(X)

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

x = tf.placeholder(tf.float32, shape=[None, 15], name='x') 
y_true = tf.placeholder(tf.float32, shape=[None,1], name='y_true')

hold_prob = tf.placeholder(tf.float32, name='hold_prob')
dropout_layer = tf.nn.dropout(x, rate=1-hold_prob)

hidden_layer1 = tf.nn.leaky_relu(layer(dropout_layer, 6))

y_hat = tf.nn.sigmoid(layer(hidden_layer1, 1), name='y_hat')

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_hat))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 1000001
batch_size=32
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    zero, one = split_data(new_X_train, y_train) 
    batch_y = np.array([[0,1]]*16).flatten()[:, np.newaxis]
    for i in range(steps):
        batch_x = prep_batch(batch_size, zero, one)
        sess.run(train, feed_dict={x:X, y_true:y, hold_prob:0.9})
    y_hat = sess.run(y_hat, feed_dict={x:new_X_test, y_true:y_test, hold_prob:1})
    saver.save(sess, "nn_model")
    print("{} file saved!".format("model.ckpt"))


print("Neural Network")
print(classification_report(y_test, np.round(y_hat)))
print(confusion_matrix(y_test, np.round(y_hat)))

