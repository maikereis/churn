# Predict Customer Churn
## Project Description

This is a small template of a project following some coding best practices.

## Files and data description

Project structure

    ├── data                                      <- Data folder.
    ├── .gitignore                                <- Git ignore file.
    ├── churn_library.py                          <- Script to perfom data exploratory analysis and modeling
    ├── churn_notebook.ipynb                      <- Notebook to perfom data exploratory analysis and modeling
    ├── churn_script_logging_and_tests.ipynb      <- Script to perfom tests on the churn_library.py
    ├── LICENSE    
    ├── pytest.ini                                <- Pytest configuration file            
    ├── README                                    <- You're reading this file       
    ├── requirements.txt                          <- All the dependencias to run this project are listed here
    
## Running Files

1. Create a new virtual enviroment with python>=3.8, also make sure of installing python3-pip:

2. Install the requirements on the environment:

    pip install requirements.txt

3. Running scrips:

    `python churn_library.py`    # or python3 churn_library.py

    After this you will see 3 new folders:


        |
        ├── images                           <- images
        |   ├── eda                          <- EDA images
        |   |   ├── churn.png                <- Churn bernoulli distribution.
        |   |   ├── customer_age.png         <- Customer age historgram.
        |   |   ├── heatmap.png              <- Correlation heatmap.
        |   |   ├── marital_status.png       <- Marital status barplot.
        |   |   └── total_trans_ct.png       <- Total transactions count distribution.
        |   |
        |   ├── results
        |   |   ├── feature_importance.png   <- Random Forest feature importances.
        |   |   └── roc_auc.png              <- Roc AUC for the test data.
        |   |
        |   └── models
        |       ├── logistic_model.pkl       <- The Logistic Regression model binary.
        |       └── rfc_model.pkl            <- The Random Forest model binary.
        |

    `pytest`    # or pytest churn_script_logging_and_tests.py or pytest churn_script_logging_and_tests.py

    After this you will see 1 new folder:


        |
        ├── logs                            <- images
        |   └── churn_library.log           <- The logfile.
        |

