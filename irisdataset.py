from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

train_data = pd.read_csv('data/iris.data')
train_data = train_data.sample(frac=1)
data = train_data.copy()
y = train_data.pop('class_name')
y = pd.get_dummies(y)

output_columns = ['Iris-setosa','Iris-versicolor','Iris-virginica']

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_data, y)

x_t =train_data[:150]
y_t =y[:150]
acc = model.score(x_t,y_t)
print(f"Accuracy of :{acc}")


X_predict = train_data[:-50]

predictions = model.predict(X_predict)
output = []
for i,j  in zip(predictions,data['class_name'][:-50]):
    submit = {'Actual': j,"Predicted":output_columns[np.argmax(i)]}
    output.append(submit)

output = pd.DataFrame(output)
output.to_csv('iris_prediction.csv', index=False)
print("Your submission was successfully saved!")
