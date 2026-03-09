### Used models:

- Random Forest
Random Forest is an ensemble learning method based on multiple decision trees trained on different subsets of the data. It is widely used for tabular datasets because it can model complex nonlinear relationships and interactions between features while being relatively robust to noise and overfitting. Additionally, Random Forest provides intrinsic measures of feature importance, making it particularly suitable for studies focused on feature selection and the evaluation of variable relevance.

- XGBoost
XGBoost is a gradient boosting algorithm that builds trees sequentially, where each new tree aims to correct the errors of the previous ones. It is known for its strong predictive performance on structured/tabular datasets and has become a standard benchmark in many machine learning competitions and research studies. XGBoost is also efficient and scalable, and it provides several mechanisms for evaluating feature importance, which makes it a valuable model for assessing the impact of selected features on predictive performance.

- DNN (MLP architecture)
A Deep Neural Network based on a Multilayer Perceptron architecture serves as a deep learning baseline in this study. MLPs consist of multiple fully connected layers that can learn complex nonlinear patterns in the data through hierarchical feature transformations. Including a DNN allows the study to compare traditional ensemble-based methods with a deep learning approach, providing insight into how feature selection affects models that learn representations through layered neural structures.


Comment on the low f1-score result in RF:
The low F1 for Random Forest (0.19) is actually consistent with findings that Random Forest performs poorly under severe class imbalance compared to XGBoost 


Comment on the MLP loss curve
the MLP is the weakest model precisely because it overfits to training data. This actually strengthens your feature selection study — irrelevant features hurt MLPs more than tree models, so you should expect the largest improvement in MLP after feature selection.

The models had overfitting so we are addressing that issue by applying some changes on the models' hyperparameters:
hyperparameters:
  mlp:
    hidden_layer_sizes: [64, 32]
    alpha: 0.01
    early_stopping: true
    validation_fraction: 0.1
    max_iter: 200
    random_state: 42
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_leaf: 10
    random_state: 42
    n_jobs: -1
  xgboost:
    max_depth: 4
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    n_estimators: 200
    random_state: 42
    n_jobs: -1
    eval_metric: "logloss"