import pandas as pd
import os

path = os.getcwd()
label_path = os.path.join(path, 'safety', 'labels', 'part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')
label_df = pd.read_csv(label_path)

features_path = os.path.join(path, 'safety', 'features')
features_files = os.listdir(features_path)
features_files.remove('.DS_Store')

features_df_list = []

for file in features_files:
    feature = os.path.join(features_path, file)
    features_df_list.append(pd.read_csv(feature))

features_df = pd.concat(features_df_list, axis=0, ignore_index=True)

# Get sample
id = pd.DataFrame(features_df['bookingID'].unique(), columns=['bookingID'])
id = id.merge(label_df, how='left', on='bookingID', suffixes=('','r'))

print('ID')

test = id.sample(frac=0.3, random_state=42)
train = pd.concat([id, test], axis=0).drop_duplicates(keep=False)

print('DF')
df = features_df.merge(label_df, how='left', on='bookingID', suffixes=('','r'))
df_test = df.merge(test, how='inner', on='bookingID', suffixes=('','r')).drop('labelr', axis=1)
df_train = df.merge(train, how='inner', on='bookingID', suffixes=('','r')).drop('labelr', axis=1)
print(df.shape)
print(df.isna().sum())

print(df_test.shape)
print(df_test.isna().sum())

print(df_train.shape)
print(df_train.isna().sum())

df.to_hdf('all_data.h5', 'grabai')
df_test.to_hdf('val_data.h5', 'grabai')
df_train.to_hdf('train_data.h5', 'grabai')
