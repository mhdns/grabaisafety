import pickle
import pandas as pd
import numpy as np
import os
import tensorflow as tf

from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

class DriverSafetyPredictor:
	def __init__(self):
		self.pca = joblib.load('pca_fit.sav')
		self.logistic = joblib.load('logistic.sav')
		self.tree = joblib.load('decision_tree.sav')
		self.forest = joblib.load('random_forest.sav')
		self.svc = joblib.load('svc.sav')
		self.knn = joblib.load('grad_boost.sav')
		self.X = None
		self.bookingID = None
		self.prediction = None
		self.report = None


	def load_data(self, pandas_dataframe):
		def grouped_data(df):
			extra_features = []
			all_cols = list(df)
			cols = all_cols
			cols.remove('bookingID')
			
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
			
			return merged_data

		def add_features(df):

			# Direction
			df['up'] = df.acceleration_y.apply(lambda y: 1 if np.round(y,2)+9.81 < 0 else 0)
			df['down'] = df.acceleration_y.apply(lambda y: 1 if np.round(y,2)+9.81 > 0 else 0)
			
			df['right'] = df.acceleration_x.apply(lambda x: 1 if np.round(x,2) < 0 else 0)
			df['left'] = df.acceleration_x.apply(lambda x: 1 if np.round(x,2) > 0 else 0)

			# Smoothness
			df['rl_smooth'] = np.where((df.Speed!=0) & (df.gyro_y!=0),
			                           df.Speed/np.abs(df.gyro_y),
			                           df.Speed + np.abs(df.gyro_y))
			
			df['ud_smooth'] = np.where((df.Speed!=0) & (df.gyro_x!=0),
			                           df.Speed/np.abs(df.gyro_x),
			                           df.Speed + np.abs(df.gyro_x))
			
			df['smoothness'] = np.sqrt(np.square(df.rl_smooth) + np.square(df.ud_smooth))
			
			# Intensity
			df['rl_intensity'] = np.where(df.acceleration_x != 0, 
			                              df.rl_smooth * np.abs(df.acceleration_x), 
			                              df.rl_smooth)
			df['ud_intensity'] = np.where(df.acceleration_x != 0, 
			                              df.ud_smooth * np.abs(df.acceleration_y), 
			                              df.ud_smooth)
			df['intensity'] = np.sqrt(np.square(df.rl_intensity) + np.square(df.ud_intensity))
			
			return df


		data = pandas_dataframe
		data.Speed.fillna(data.Speed.median(), inplace=True)
		data = add_features(data)

		cols = list(data)
		cols.remove('bookingID')
		
		scaler = Normalizer()
		scaler = scaler.fit(data[cols])
		data[cols] = scaler.transform(data[cols])

		data = grouped_data(data)

		cols = list(data)
		scaler2 = Normalizer()
		scaler2 = scaler2.fit(data)
		data[cols] = scaler2.transform(data[cols])
		data = data.reset_index()

		self.bookingID = data.iloc[:, 0:1]
		self.X = self.pca.transform(data.iloc[:, 1:])

		def get_mod(array, models):
			final_model = array
			for i in models:
				final_model = np.concatenate((final_model, i.predict_proba(array)[:, 0:1]), axis=1)
		
			return final_model

		try:
			self.X = get_mod(self.X, models=[self.logistic, self.tree, self.forest, self.svc, self.knn])
		except:
			self.backup()
			self.X = get_mod(self.X, models=[self.logistic, self.tree, self.forest, self.svc, self.knn])

		print("Data Loaded and Transformed!")
		return None

	def predict(self):
		with tf.Session() as sess:
			saver = tf.train.import_meta_graph('nn_model.meta')
			saver.restore(sess,tf.train.latest_checkpoint('./'))
			graph = tf.get_default_graph()
			x = graph.get_tensor_by_name("x:0")
			hold_prob = graph.get_tensor_by_name("hold_prob:0")
			y_hat = graph.get_tensor_by_name("y_hat:0")

			y_hat = sess.run(y_hat, feed_dict={x: self.X, hold_prob: 1})


		self.prediction = np.round(y_hat)
		pred_series = pd.Series(self.prediction[:,0], name="Prediction")
		self.report = pd.concat([self.bookingID, pred_series], axis=1)
		return None

	def backup(self):
		all_data = pd.read_hdf('data_final.h5', 'grabai')
		X = all_data.iloc[:, 1:-1]
		y = all_data.iloc[:, -1]
		pca = PCA(n_components=10)
		X = pca.fit_transform(X)
		y = np.array(y)[:, np.newaxis]

		self.logistic = LogisticRegression(C=10000).fit(X, y)
		self.tree = DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=100).fit(X, y)
		self.forest = RandomForestClassifier(criterion='entropy', n_estimators=10).fit(X, y)
		self.svc = SVC(probability=True).fit(X, y)
		self.knn = KNeighborsClassifier(n_neighbors=15).fit(X, y)

		print('Contingency Complete')
		return None




