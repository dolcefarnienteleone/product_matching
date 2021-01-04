### product_matching

#Overview: 

All the required datasets for running the project is included. The python code brief explanation is in below.

#create_corpus_from_train.py

This file reads the initial training JSON data into text corpus for doing following the embedding process.
The output file name using in the project is - train_corpus_v4.txt

#embed.py

This runs the word2vec embedding process to have all words embedded with vector. 
The output file name using in the project is - w2v_cbow_v3.model or w2v_sg_v3.model

#create_df.py

This file reads the initial JSON file into pandas dataframe for further processing.
The order of attributes in JSON file of training and validation set is different from the hidden testing set. The index to retrieve information for the attribute is adapted from training and validation to testing through comment out.

The output file names using in the project are:
1. trainset_df_t10.csv
2. validset_df_t10.csv
3. testset_df_t10.csv

#product_vec_classifier_exp1.py

This is the 1st experiment we set for running by using initial training, validation, and testing sets to do training, validation, and testing respectively. The output of this experiment result includes Table 1 & 2 in the report, which is the comparison table for the cross-validation of each model and the accuracy of validation and testing scores. 

Additionally, some of the comments out lines are for the different combination of left and right product vectors that we’ve tried. Such as by summation of all attributes’ vector to get just size of 25 for one product or the np.subtract.

#product_vec_classifier_exp2.py

Most of the code is same as experiment 1, however, we combine the initial training and validation set together and do the train_valid_split by the ratio of 70:30 to enhance the training data. Testing set is still the original testing. The result is Table 3 in the report. 

We also add the grid search function from Random Forest and Neural Networks (MLP) models by importing the function from Gridsearch_fun.py file. The grid search takes some time to process, especially the RF one. The comparison table running is commented out since it’s not necessary to run here.
Once gaining the best parameter, can use the best parameters (which are commented out in line 270 and 283) to rerun the validation and testing score to see the improvement. The output is in Table 4 in the project report.

#Gridsearch_fun.py

There are two grid search functions in this file: Random Forest (RF_GridSearch) another for Neural Networks (MLP_GridSearch). The different grid will be created respectively, depending on the parameters that can be adjusted according to the sklearn classifier function. Fit in the X (attributes) and y (target) to get_bestparam process to gain the best parameters from the grid search. 

#subset_testing.py

Running this file provides the input for experiment 3. We subset the original testing set into three sets. The positive and negative samples are equally distributed, resulting in the ratio shall be the same as the original. 

The output file names using in the project are:
1. testset_df_t13_1.csv
2. testset_df_t13_2.csv
3. testset_df_t13_3.csv

#product_vec_classifier_exp3.py

This includes the code to run our experiment no.3, using a combination of original training, validation, and the subset of testing to get the testing classification score. Both RF and MLP models utilize the best parameters that we gained from product_vec_classifier_exp2.py. There are three subsets of the testing set as input at the beginning, which is the output from subset_testing.py. We also use commenting out to change the subsets. The results are Table 5, 6, & 7 in the report.

#product_vec_classifier_exp4.py

The code is for running experiment no.4, combining training, validation, and testing sets as the new df to do the train_valid split and check the testing result. The result is Table 8 in the report.

#embed_more_prepro.py

With more preprocess function, the rest of the code is the same as embed.py. For running the experiment for adding more preprocess. 
The embedded model in this project is: w2v_cbow_v4.model or w2v_sg_v4.model

#subset_training.py

The processing time for both “adding more preprocess” and “adding KG vectors” are enormous. We use a subset of the original training set to do a small portion trial run, which is 5% of the actual training set. 
The output file is using in the project: trainset_df_t13.csv

#product_vec_classifier_more_prepro.py

Adding more preprocess into original model to experiment. Most of the code is similar to the product_vec_classifier_expX.py, only adding more preprocess.
The result is Table 9 in the report. 

#product_vec_classifier_withKGvec.py

Most of the code is similar to the product_vec_classifier_exp.py series of the file, except that we add spacy_ent_to_vec function to enable the system with retrieving KG vector through online API services. The result is in Table 10 for the testing scores, and the time cost in Table 11 by using a subset of actual training.

The requirements for running this code: 
pip install -U spacy
python -m spacy download en_core_web_sm

#kg/Lookup.py

The function to do entity query from online pre-computed KG, e.g. DBpedia. The code is slightly changed from the original source code, reference link in below, to get the entity information only without other unnecessary information.

Ref. https://github.com/ernestojimenezruiz/tabular-data-semantics-py/blob/master/TabularSemantics/src/kg/lookup.py






