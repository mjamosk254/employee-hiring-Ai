import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#employee features

#[experience,interview score,communication]

x=np.array([
    [1,55,50],
    [2,60,58],
    [3,65,62],
    [4,72,70],
    [5,80,78],
    [6,88,85],
    [7,92,90],
    [8,95,94],
    [2,58,54],
    [5,82,80],
    [6,85,83],
    [7,91,88]
])

#o-rejected
#1-accepted

y=np.array([
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    1
])

#split the dataset
x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

#build the neural network
model=tf.keras.Sequential([
    tf.keras.layers.Dense(8,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

#compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#train the model
model.fit(x_train,y_train,epochs=150)

#evalute the model
loss,accuracy=model.evaluate(x_test,y_test)

print("loss: ", loss)
print("Accuracy: ", accuracy)

#prediction
new_employee=np.array([
    [6,90,88]
])

prediction=model.predict(new_employee)
print("prediction: ", prediction)

#predict every employee in thr testing data
predictions=model.predict(x_test)
predicted_labels=(predictions >=0.5).astype(int)

#display the actual answers and model's prediction
print("Actual labels")
print(y_test)

print("predicted labels")
print(predicted_labels) 

#save the trained model
model.save("employee_hiring_model.keras")
print("Model saved successfully")