from keras.datasets import mnist

def write_data_set_csv(X, y, file_name, N_samples):
    if not N_samples:
        N_samples = X.shape[0]
    print(f"file: {file_name} will contain {N_samples}...")
    with open(file_name, "w") as f:
        for n in range(N_samples):
            line = ""
            for i in range(X.shape[1]):
                for j in range(X.shape[2]):
                    line += str(round(X[n, i, j] / 255, 4)) + ","
            line += str(y[n]) + "\n"
            #print(line)
            f.write(line)

#loading
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#shape of dataset
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
 
#writing csv for the data base
write_data_set_csv(train_X, train_y, "training_data.txt", 5000)
write_data_set_csv(test_X, test_y, "test_data.txt", 1000)

#plotting
from matplotlib import pyplot
for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()