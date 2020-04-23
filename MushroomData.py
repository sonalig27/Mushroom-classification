import pandas as pd
df = pd.read_table('agaricus-lepiota.data', delimiter=',', header=None)
####df.head()
####df.info()
column_labels = [
    'class', 'cap shape', 'cap surface', 'cap color', 'bruised', 'odor',
    'gill attachment', 'gill spacing', 'gill size', 'gill color',
    'stalk shape', 'stalk root', 'stalk surface above ring',
    'stalk surface below ring', 'stalk color above ring',
    'stalk color below ring', 'veil type', 'veil color', 'ring number',
    'ring type', 'spore print color', 'population', 'habitat'
]

df.columns = column_labels
#####df = df[df['stalk root'] != '?']
X = df.loc[:]
X_enc = pd.get_dummies(X)
X_enc.to_csv('dataset.txt', index = None, sep = ',',mode='a')
total_records = len(X_enc.index)
train = X_enc.loc[0:(total_records/2)-1]
print(train.shape)
val = X_enc.loc[(total_records/2):((total_records * 3)/4)-1]
print(val.shape)
test = X_enc.loc[(total_records * 0.75):total_records]
print(test.shape)
train.to_csv('training.txt',header = None, index = None, sep = ',',mode='a')
val.to_csv('val.txt',header = None, index = None, sep = ',',mode='a')
test.to_csv('testing.txt',header = None, index = None, sep = ',',mode='a')

