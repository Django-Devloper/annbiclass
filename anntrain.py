import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder ,LabelEncoder
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf 

data = pd.read_csv('CVD_cleaned.csv')

def pickel_saver(colunm_name , encoder):
    with open(colunm_name+'encoder.pkl' , 'wb') as file:
      pickle.dump(encoder ,file)

def lable_encoder(column_name ,data):
    lable_encoder = LabelEncoder()
    data[column_name] = lable_encoder.fit_transform(data[column_name])
    pickel_saver(column_name,lable_encoder)
    return data


def one_hot_encoder(column_name ,data):
    hot_encoder = OneHotEncoder()
    column_encoded = hot_encoder.fit_transform(data[[column_name]])
    colunm_encoded_df = pd.DataFrame(column_encoded.toarray() , columns=hot_encoder.get_feature_names_out([column_name]))
    data = pd.concat([data.drop(column_name,axis=1),colunm_encoded_df],axis=1)
    pickel_saver(column_name,hot_encoder)
    return data

data = lable_encoder('Exercise' ,data)
data = lable_encoder('Heart_Disease' ,data)
data = lable_encoder('Skin_Cancer' ,data)
data = lable_encoder('Other_Cancer' ,data)
data = lable_encoder('Depression' ,data)
data = lable_encoder('Arthritis' ,data)
data = lable_encoder('Sex' ,data)
data = lable_encoder('Smoking_History' ,data)

data = one_hot_encoder('General_Health' , data)
data = one_hot_encoder('Checkup' , data)
data = one_hot_encoder('Diabetes' , data)
data = one_hot_encoder('Age_Category' , data)

x = data.drop('Heart_Disease',axis=1)
y = data['Heart_Disease']

x_train , x_test , y_train ,y_test = train_test_split(x,y , train_size=0.8, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(16 , activation ='relu'),
    Dense(8 , activation = 'relu'),
    Dense(1 , activation = 'sigmoid')
])

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt , loss='binary_crossentropy' , metrics=['accuracy'])
early_stopping_callbck = EarlyStopping(monitor = 'val_loss' ,patience=5, restore_best_weights=True)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                    epochs=100, callbacks=[early_stopping_callbck])
model.save('ann_binary_classification.keras')