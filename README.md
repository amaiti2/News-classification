The code in this repository is designed for the classification of the news articles based on their headline and short description. The data was obtained from Rishabh Misra's Kaggle dataset [1][2] of Huffpost articles and corresponding labels at the following link: https://www.kaggle.com/datasets/rmisra/news-category-dataset 

Rather than attempting classification on all 42 categories at once, we selected 12 of the larger categories to classify among. 

1) PREPROCESSING: To preprocess the data and generate training, validation, and test sets, first run the Data_cleaning.ipynb notebook, which will remove puncuation, stop words, and perform lemmatization to the headline and short descriptions, outputting the file 'News_ds_cleaned.pkl', then run 'Reduce_df_&_train_test_split.ipynb' to finish other aspects of the preprocessing and perform the train/validation/test split, and output the corresponding files. 

There were 3 different model choices attempted: logistic regression (79% validation accuracy), boosted decision tree (xgboost) (61% validation accuracy), and a convolutional neural net (82% validation accuracy). 

2) TFIDF MODELS: The first two require the data be converted to tf-idf format, which then has its dimension reduced with PCA first before the model can be applied. However, the CNN can be applied directly without further preprocessing at this point. To do logisitic regression or XGBoost, first run 'TFIDF_&_PCA_for_training_&_validation_data.ipynb' as well as 'TFIDF_&_PCA_for_test_data.ipynb' to transform the test data. These notebooks have large computational and memory requirements, and so we needed to run them on google colab with GPU and extra memory; they are set up for that and some lines may need to be modified if they are to be run off of colab. Once the 'ttest_PCA_cols.pkl' and 'tvalid_PCA_cols.pkl' are obtained, one can run the 'Logistic_regression.ipynb' and/or 'Boosted_decision_tree.ipynb' to check the performance of these models. Note that these notebooks were also set up to run on colab

3) CNN MODEL: To run the CNN model, simply run the 'CNN_model.ipynb' notebook after running the preprocessing steps outlined in 1)

Since the CNN model obtained the highest validation/test accuracy (as well as being substantially lighter and faster to train, with lower memory requirements), it is the model we endorse as our best choice of classification model for this problem. The CNN model had 81.88% classification accuracy on the test set, and a weighted average F1 score of 82%. 


[1] Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022).
[2] Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).
