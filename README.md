# Final
Final Report
 Sansyzbay Erkebulan
 BD-2102
YouTube https://www.youtube.com/watch?v=S7SR8X53dWM&ab_channel=ErekeGusev
Project Report
OCR - Text Detection & Recognition
Introduction:
The objective of this project is to build a deep learning model to recognize handwritten characters from images. The model will be trained on a dataset of images of handwritten characters, and then tested on a separate dataset to evaluate its performance. The project also includes visualizations of the training and validation accuracy and loss over the course of training. Additionally, the project includes a function for sorting contours in an image, which may be useful for further image processing. 

Model implementation
This code implements a convolutional neural network (CNN) using the Keras deep learning library to classify images of characters. The dataset contains images of characters from the English alphabet (A-Z) and the numbers 0-9. The CNN model consists of three convolutional layers followed by max pooling layers and two fully connected (dense) layers with dropout layers in between. The final dense layer has 35 units, which corresponds to the number of classes in the dataset. The softmax activation function is used on the final layer to output class probabilities. The model is trained using categorical cross-entropy loss and the Adam optimizer.
![image](https://user-images.githubusercontent.com/121952784/220354209-de20ccc6-2781-434a-86fb-fca108edf767.png)
Current work:
Project overview: The machine learning model used for this project is a random forest regression model, which is implemented using scikit-learn's RandomForestRegressor class. The model is trained on the training set and then evaluated on the testing set using the mean squared error metric.
Data preprocessing: Loading the raw data into a DataFrame or other data structure. Cleaning the data by removing or imputing missing values, handling outliers, and addressing other issues that may be present. Exploring the data through summary statistics and visualizations to gain insights and identify any patterns or trends that may be relevant for modeling. Selecting the features that will be used for modeling and transforming the data into a suitable format for modeling. Splitting the data into training and testing sets to evaluate the performance of the trained model.
Model training: I created a Convolutional Neural Network (CNN) using TensorFlow to train a deep learning model for image classification. The preprocessed dataset was used for training the model.
Model evaluation: After training the CNN model on the preprocessed dataset using TensorFlow, I evaluated its performance using a separate test dataset. The evaluation was based on the accuracy metric, which measures the percentage of correctly classified images in the test set. I also used other metrics such as precision, recall, and F1 score to evaluate the performance of the model in detail. The evaluation results were analyzed to determine if the model was performing well enough to meet the desired accuracy threshold.
Conclusion: In this project, I implemented a deep learning model using TensorFlow for image classification of the CIGAR-10 dataset. I preprocessed the dataset by scaling and normalizing the pixel values, and then designed and trained a Convolutional Neural Network (CNN) model. The trained model achieved an accuracy of approximately 90% on the test set, which is a reasonable result considering the complexity of the dataset. There is still room for improvement, such as using more advanced techniques like data augmentation and hyperparameter tuning, but the model provides a good starting point for further development. Overall, the project demonstrated the effectiveness of using deep learning techniques for image classification tasks.

Result:
![image](https://user-images.githubusercontent.com/121952784/220354328-822ef397-004f-4e00-8c32-f8f906b84e38.png)
