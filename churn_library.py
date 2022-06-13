"""Module to import, perform EDA and build features."""
# pylint: disable=W0621

import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings("ignore")


# import libraries

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (20, 10)

DATA_PATH = Path("./data/bank_data.csv")
EDA_IMAGES_PATH = Path("./images/eda")
MODELS_PATH = Path("./models")
RESULTS_PATH = Path("./images/results")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

CAT_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

NUM_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

KEEP_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]


def import_data(file_path: Path):
    """
    Return a dataframe for the csv found at pth.

    input:
            file_path: a path to the csv
    output:
            data: pandas dataframe
    """
    if not file_path.is_file():
        raise FileNotFoundError("The path isn't a valid path to a file")

    data = pd.read_csv(file_path.resolve(), index_col=0)
    return data


def perform_eda(dataset):
    """
    Perform eda on df and save figures to images folder.

    input:
            dataset: pandas dataframe

    output:
            None
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(
            "The dataset passed as an argument is not a pandas' DataFrame")

    # Plotting some info about the dataset
    print(f"A taste of the data:\n{dataset.head()}")
    print(f"\nShape of the dataset:\n{dataset.shape}")
    print(f"\nMissing values:\n{dataset.isnull().sum()}")
    print(f"\nSome descritive measures:\n{dataset.describe()}")

    # Create the 'images' folder if it doesn't exist.
    os.makedirs(EDA_IMAGES_PATH, exist_ok=True)

    # Plot the churn bernoulli distribution
    sns.histplot(x=dataset["Churn"])
    # Save figure
    plt.savefig(EDA_IMAGES_PATH / "churn.png")
    # Clean plot
    plt.clf()

    # Plotting and saving the the customer Customer_Age histplot
    sns.histplot(x=dataset["Customer_Age"])
    plt.savefig(EDA_IMAGES_PATH / "customer_age.png")
    plt.clf()

    # Plotting and saving the the customer Marital_Status countplot
    sns.countplot(x=dataset["Marital_Status"])
    plt.savefig(EDA_IMAGES_PATH / "marital_status.png")
    plt.clf()

    # Plotting and saving the the customer Total_Trans_Ct histplot
    sns.histplot(x=dataset["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig(EDA_IMAGES_PATH / "total_trans_ct.png")
    plt.clf()

    # Plotting and saving the the correlation heatmap
    sns.heatmap(dataset.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.tight_layout()
    plt.savefig(EDA_IMAGES_PATH / "heatmap.png")
    plt.clf()


def encoder_helper(dataset, category_lst, target):
    """
    Create new columns using the with the mean churn \
    for each category in the category list.

    input:
            dataset: pandas dataframe
            category_lst: list of columns that contain categorical features
            target: string of response name.

    output:
            dataset: pandas dataframe with new columns
    """
    encoder = TargetEncoder()

    # Create a column with the mean churn in each category in the list.
    for category in category_lst:

        dataset[category + "_Churn"] = encoder.fit_transform(
            dataset[category], dataset[target]
        )

    return dataset


def perform_feature_engineering(dataset, target):
    """
    Perfom the feature engineering.

    input:
              dataset: pandas dataframe
              target: string of response name.
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    # Create the encoded features
    features = encoder_helper(dataset, CAT_COLUMNS, target)

    features, targets = features.drop(target, axis=1), features[target]
    features = features[KEEP_COLUMNS]

    # Return X_train, X_test, y_train, y_test datasets
    return train_test_split(features, targets, test_size=0.3, random_state=42)


def classification_report_image(
    rf_clf,
    lr_clf,
    x_train,
    x_test,
    y_train,
    y_test,
):
    """
    Produce classification reports for training and testing results\
    and stores report as image in images folder.

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    y_test_preds_rf = rf_clf.predict(x_test)
    y_test_proba_preds_rf = rf_clf.predict_proba(x_test)[:, 1]
    y_train_preds_rf = rf_clf.predict(x_train)

    y_test_preds_lr = lr_clf.predict(x_test)
    y_test_proba_preds_lr = lr_clf.predict_proba(x_test)[:, 1]
    y_train_preds_lr = lr_clf.predict(x_train)

    try:
        # scores
        print("RandomForest results:\n")
        print("test results:\n")
        print(classification_report(y_test, y_test_preds_rf))
        print("train results:\n")
        print(classification_report(y_train, y_train_preds_rf))

        print("LogisticRegression results:\n")
        print("test results:\n")
        print(classification_report(y_test, y_test_preds_lr))
        print("train results:\n")
        print(classification_report(y_train, y_train_preds_lr))

        _, axis = plt.subplots(figsize=(15, 8))
        RocCurveDisplay.from_predictions(
            y_test, y_test_proba_preds_rf, name="RandomForest", ax=axis
        )
        RocCurveDisplay.from_predictions(
            y_test, y_test_proba_preds_lr, name="LogisticRegression", ax=axis
        )
        plt.savefig(RESULTS_PATH / "roc_auc.png")
        plt.clf()

    except ValueError as err:
        raise err


def feature_importance_plot(model, train_features, output_pth):
    """
    Create and stores the feature importances in pth.

    input:
            model: model object containing feature_importances_
            train_features: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    try:
        importances = pd.Series(
            model.best_estimator_.feature_importances_,
            index=train_features.columns)
    except ValueError as err:
        raise err

    importances_plot = sns.barplot(x=importances.index, y=importances.values)
    importances_plot.set_title("Feature Importance")
    importances_plot.set_ylabel("Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.clf()


def train_models(x_train, x_test, y_train, y_test):
    """
    Train and store model results: images + scores, and store models.

    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42, max_features="sqrt")
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [10, 50],
        "max_features": ["sqrt"],
        "max_depth": [4, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    # Find best estimator given the set of parameters
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Train both models
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_test, y_test)

    # Create the folder if doesn't exists.
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Store models
    joblib.dump(cv_rfc.best_estimator_, MODELS_PATH / "rfc_model.pkl")
    joblib.dump(lrc, MODELS_PATH / "logistic_model.pkl")

    print("Plotting..")
    classification_report_image(
        cv_rfc.best_estimator_,
        lrc,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    feature_importance_plot(
        cv_rfc,
        x_train,
        RESULTS_PATH /
        "feature_importance.png")


def create_churn_flag(dataset):
    """
    Create a Churn columns based on the Attrition_Flag.

    input:
              dataset: the churn dataset
    output:
              dataset: dataset with the churn column
    """
    # Create the churn feature
    dataset["Churn"] = dataset["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return dataset


if __name__ == "__main__":

    churn_data = import_data(DATA_PATH)
    churn_data = create_churn_flag(churn_data)

    perform_eda(churn_data)

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        churn_data, target="Churn"
    )

    train_models(X_train, X_test, y_train, y_test)
