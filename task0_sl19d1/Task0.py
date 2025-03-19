
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Read CSV file into a NumPy array (matrix)
matrix = np.loadtxt('Task0/task0_sl19d1/test.csv', delimiter=',', skiprows=1)

# Separate y from x1-x10 in train
#y = matrix[:, 1]
#x = matrix[:,2:]

# Separate y from x1-x10 in test
x = matrix[:,1:]
ones = np.ones(np.shape(x)[1])
y_pred = np.dot(x, ones)/len(ones)

#check if output near y in train
#RMSE = mean_squared_error(y, y_pred)**0.5
#print(RMSE)

# Create the index column
ids = np.arange(len(y_pred)) + 10000

# Create a DataFrame with 'Id' and 'y_pred' columns
submission_df = pd.DataFrame({
    'Id': ids,
    'y': y_pred
})

# Write the DataFrame to a CSV file
submission_df.to_csv('Task0/task0_sl19d1/submission_task0.csv', index=False)






