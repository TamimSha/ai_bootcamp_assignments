import matplotlib.pyplot as plt



# import dataset, classifiers and performance matrices

from sklearn import datasets, svm, metrics

from sklearn.model_selection import train_test_split



digits = datasets.load_digits()

# getting the data

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))

for ax, image, label in zip(axes, digits.images, digits.target):

  ax.set_axis_off()

  ax.imshow(image, cmap=plt.cm.gray_r, interpolation ='nearest')

  ax.set_title('Training: %i' % label)



# we need to flatten the images

n_samples =len(digits.images)

data = digits.images.reshape((n_samples, -1))



#create classifier > Svc

clf = svm.SVC(gamma =0.001)



#split the data into 50% train and 50% for test

x_train, x_test, y_train, y_test = train_test_split(

  data, digits.target, test_size=0.5, shuffle=False)



# learn the dataset

clf.fit(x_train, y_train)



predicted =clf.predict(x_test)



# prediction of you

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))

for ax, image, label in zip(axes, digits.images, digits.target):

  ax.set_axis_off()

  image = image.reshape(8,8)

  ax.imshow(image, cmap=plt.cm.gray_r, interpolation ='nearest')

  ax.set_title(f'Prediciton : (prediction)')



print(f"Classification report for classifier (clf):\n"

   f"(metrics.classification_report(y_test, predicted))\n")



disp = metrics.plot_confusion_matrix(clf, x_test, y_test)

disp.figure_.suptitle("Confusion Matrix")



print(f"Confusion matrix:\n(disp.confusion_matrix)")

plt.show()