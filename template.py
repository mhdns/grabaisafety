from predictor import DriverSafetyPredictor
import pandas as pd

data = pd.read_hdf('val_data.h5', 'grabai')
data = data.drop('label', axis=1)
data = data.head(2000)
predictor = DriverSafetyPredictor()
predictor.load_data(data)
print(predictor.X)

predictor.predict()

print(predictor.report)