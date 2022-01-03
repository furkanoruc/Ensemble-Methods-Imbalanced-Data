def data_balancer(X, y, class_1, class_2, ratio_1, ratio_2):
    import pandas as pd
    ratio_2 = 1-ratio_1

    augmented = pd.concat([pd.DataFrame(X),pd.DataFrame(y, columns=['label'])],axis=1)
    augmented_filtered = augmented[(augmented['label']==class_1) | (augmented['label']==class_2)]
    augmented_filtered.reset_index()
    num_class_1 = augmented_filtered[augmented_filtered['label'] == class_1].shape[0]
    num_class_2 = augmented_filtered[augmented_filtered['label'] == class_2].shape[0]
    #if num_class_2>num_class_1:
    if (ratio_2/ratio_1) > (num_class_2/num_class_1):
        expected_num_class_1 = round(num_class_2/(ratio_2/ratio_1),0)
        expected_num_class_2 = num_class_2
    else:
        expected_num_class_1 = num_class_1
        expected_num_class_2 = round(num_class_1/(ratio_1/ratio_2),0)
    """
    else:
        if (ratio_2/ratio_1) > (num_class_2/num_class_1):
            expected_num_class_1 = round(num_class_2/(ratio_2/ratio_1),0)
        else:
            expected_num_class_2 = round(num_class_1/(ratio_1/ratio_2),0)
    """
    drop_num_class_1 = int(num_class_1 - expected_num_class_1)
    drop_num_class_2 = int(num_class_2 - expected_num_class_2)

    augmented_filtered.drop(augmented_filtered[augmented_filtered['label']==class_1].sample(n=drop_num_class_1).index,
    axis=0, inplace=True)
    augmented_filtered.drop(augmented_filtered[augmented_filtered['label']==class_2].sample(n=drop_num_class_2).index,
    axis=0, inplace=True)

    num_class_1_after_balance = augmented_filtered[augmented_filtered['label'] == class_1].shape[0]
    num_class_2_after_balance = augmented_filtered[augmented_filtered['label'] == class_2].shape[0]

    y_filtered = augmented_filtered['label'].to_numpy()
    augmented_filtered.drop(labels='label', axis=1, inplace=True)
    X_filtered = augmented_filtered.to_numpy()
    Balance = (num_class_1/(num_class_1+num_class_2), num_class_2/(num_class_1+num_class_2))
    Balance_After_Operation = (round(num_class_1_after_balance/(num_class_1_after_balance+num_class_2_after_balance),2),
    round(num_class_2_after_balance/(num_class_1_after_balance+num_class_2_after_balance),2))
    return X_filtered, y_filtered, Balance, Balance_After_Operation

def MultiLayerPerceptron(X, y, X_test, y_test, hidden_layer_units=25, alpha=0.03, epoch=20):
    import numpy as np
    import pandas as pd
    import time
    # Initialization
    #X = tr_images
    #y = tr_labels
    y_reserved_train = y
    y = pd.get_dummies(y).to_numpy()
    #X_test = tt_images
    #y_test = tt_labels
    y_reserved_test = y_test
    y_test = pd.get_dummies(y_test).to_numpy()

    #hidden_layer_units = 25

    W1 = Weight_Initialization(X.shape[1], hidden_layer_units, -0.01, 0.01)
    W2 = Weight_Initialization(hidden_layer_units, y.shape[1], -0.01, 0.01)
    b1 = Weight_Initialization(1, hidden_layer_units, -0.01, 0.01)
    b2 = Weight_Initialization(1, y.shape[1], -0.01, 0.01)

    #alpha = 0.03
    #epoch = 20

    Accuracy_train_accumulated = []
    Accuracy_test_accumulated = []
    # Forward Pass

    for j in range(epoch):
        start = time.time()
        for i in range(X.shape[0]):
            Z1 = np.dot(X[i],W1) + b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1,W2) + b2
            A2 = softmax(Z2)

            # Changed for Different Digit Duals
            if np.argmax(A2,axis=1)==0:
                classes=3
            else:
                classes=8
            #classes = np.argmax(A2,axis=1)
            #Accuracy = (sum(classes == y_reserved_train[i]))
            Accuracy = classes == y_reserved_train[i]
            Loss = Loss_Perceptron(y[i],A2)
            
            W1_Update = alpha * np.dot(X[i].T.reshape(X.shape[1],1), np.multiply((np.dot(W2, (y[i]-A2).T)).T, A1 - np.power(A1, 2)))
            b1_Update = alpha * np.sum(np.multiply(np.dot(W2, (y[i]-A2).T).T, A1 - np.power(A1, 2)), axis = 1, keepdims = True)

            W2_Update = alpha * np.dot(A1.T, (y[i]-A2))
            b2_Update = alpha * np.sum((y[i]-A2), axis = 1, keepdims = True)

            W1 += W1_Update
            W2 += W2_Update
            b1 += b1_Update
            b2 += b2_Update
        
        # Train Set Accuracy for Each Epoch
        highest_weight_train = []
        highest_bias_train = []
        highest_accuracy_train = 0

        Z1 = np.dot(X,W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1,W2) + b2
        A2 = softmax(Z2)

        classes_train = np.argmax(A2,axis=1)
        # Changed for Different Digit Duals
        classes_train[classes_train == 0] = 0
        classes_train[classes_train == 1] = 6

        Accuracy_train = (sum(classes_train == y_reserved_train))/X.shape[0]
        if Accuracy_train>highest_accuracy_train:
            highest_weight_train = W1
            highest_bias_train = b1
            highest_accuracy_train = Accuracy_train
        Accuracy_train_accumulated.append(Accuracy_train)
        
        # Test Set Accuracy for Each Epoch

        Z1_test = np.dot(X_test,W1) + b1
        A1_test = sigmoid(Z1_test)
        Z2_test = np.dot(A1_test,W2) + b2
        A2_test = softmax(Z2_test)
        
        classes_test = np.argmax(A2_test,axis=1)
        # Changed for Different Digit Duals
        classes_test[classes_test == 0] = 0
        classes_test[classes_test == 1] = 6

        Accuracy_test = (sum(classes_test == y_reserved_test))/X_test.shape[0]    
        Accuracy_test_accumulated.append(Accuracy_test)
        stop = time.time() 

        if j%10==0 or j==epoch-1:
            print(f'for Epoch {j}, Training Accuracy: {Accuracy_train:.3f} & Testing Accuracy: {Accuracy_test:.3f}, Time per Epoch: {(stop-start):.2f} seconds')

    return Accuracy_train_accumulated, Accuracy_test_accumulated, W1, W2, b1, b2, classes_train, classes_test

def softmax(z):
    import numpy as np
    return np.exp(z)/(np.exp(z).sum(axis=1)[:, np.newaxis])

def sigmoid(z):
    import numpy as np
    return 1/(1+np.exp(-z))

def Loss_Perceptron(y_true, y_pred):
    import numpy as np
    return -(np.multiply(y_true, np.log(y_pred)).sum())/y_true.shape[0]

def Weight_Initialization(row_number, column_number, interval_min, interval_max):
    import numpy as np
    return np.random.uniform(interval_min, interval_max, (row_number, column_number))

def KNeighbors(X, y, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, plot_confusion_matrix, auc
    import matplotlib.pyplot as plt
    accuracy_score_holder = []
    misclassified_sample_count = []
    precision_score_holder = []
    roc_auc_score_holder = []
    for i in range (1,11):
        knn=KNeighborsClassifier(n_neighbors=i, p=2, metric='minkowski') 
        knn.fit(X, y)
        myprediction_knn = knn.predict(X_test)
        accuracy_score_holder.append(accuracy_score(y_test, myprediction_knn))
        misclassified_sample_count.append((y_test != myprediction_knn).sum())
        roc_auc_score_holder.append(roc_auc_score(y_test, myprediction_knn))
        #precision_score_holder.append(y_test, knn.predict(X_test), average = 'micro')
        i = i + 1
    #print("Accuracy Score:", accuracy_score_holder)
    #print("Precision Score:", precision_score_holder)
    #print("Misclassified samples:", misclassified_sample_count)
    #print("ROC - AUC Score:", roc_auc_score_holder)
    #plot_confusion_matrix(knn, X_test, y_test)
    #plt.show()
    
    #y_scores = knn.predict_proba(X_test)
    #fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1], pos_label = 6)
    #roc_auc = auc(fpr, tpr)

    #plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    #plt.legend(loc = 'lower right')
    #plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate')
    #plt.title('ROC Curve of kNN')
    #plt.show()
    
    return accuracy_score_holder[0], roc_auc_score_holder[0], misclassified_sample_count[0]

def forest(X, y, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt
    myforest = RandomForestClassifier(criterion = 'entropy', n_estimators = 10, random_state = 60)
    myforest.fit(X, y)
    myprediction_forest = myforest.predict(X_test)
    #print("Accuracy Score: ", accuracy_score(y_test,myprediction_forest))
    #print("Number of Misclassified Samples: ", (y_test != myprediction_forest).sum())
    #print("ROC - AUC Score:", roc_auc_score(y_test, myprediction_forest))

    #plot_confusion_matrix(myforest, X_test, y_test)
    #plt.show()
    
    #y_scores = myforest.predict_proba(X_test)
    #fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1], pos_label = 6)
    #roc_auc = auc(fpr, tpr)

    #plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    #plt.legend(loc = 'lower right')
    #plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate')
    #plt.title('ROC Curve of Random Forest')
    #plt.show()
    
    return accuracy_score(y_test,myprediction_forest), roc_auc_score(y_test, myprediction_forest), (y_test != myprediction_forest).sum()

def svm(X, y, X_test, y_test):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt

    mysvm = SVC(kernel = 'poly', C = 10, random_state =60, probability = True)
    mysvm.fit(X,y)
    mysvm_prediction = mysvm.predict(X_test)
    #print("Accuracy Score: ", accuracy_score(y_test,mysvm_prediction))
    #print("Number of Misclassified Samples: ", (y_test != mysvm_prediction).sum())
    #print("ROC - AUC Score:", roc_auc_score(y_test, mysvm_prediction))

    #plot_confusion_matrix(mysvm, X_test, y_test)
    #plt.show()
    
    #y_scores = mysvm.predict_proba(X_test)
    #fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1], pos_label = 6)
    #roc_auc = auc(fpr, tpr)

    #plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    #plt.legend(loc = 'lower right')
    #plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate')
    #plt.title('ROC Curve of Support Vector Machine')
    #plt.show()
    
    return accuracy_score(y_test,mysvm_prediction), roc_auc_score(y_test, mysvm_prediction), (y_test != mysvm_prediction).sum()

def perceptron(X, y, X_test, y_test):
    from sklearn.linear_model import Perceptron
    from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt

    for i in range (30,31):
        myperceptron = Perceptron(penalty = 'elasticnet', max_iter = i, eta0 = 0.001, random_state = 60)
        myperceptron.fit(X, y)
        myprediction_perceptron = myperceptron.predict(X_test)
        #print(accuracy_score(y_test,myprediction_perceptron))
        
        #print("Accuracy Score: ", accuracy_score(y_test,myprediction_perceptron))
        #print("Number of Misclassified Samples: ", (y_test != myprediction_perceptron).sum())
        #print("ROC - AUC Score:", roc_auc_score(y_test, myprediction_perceptron))

    #plot_confusion_matrix(myperceptron, X_test, y_test)
    #plt.show()
    
    return accuracy_score(y_test,myprediction_perceptron), roc_auc_score(y_test, myprediction_perceptron), (y_test != myprediction_perceptron).sum()