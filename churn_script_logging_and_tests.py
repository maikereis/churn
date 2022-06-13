"""Tests for the churn_library.py."""
import logging
import warnings

import pandas as pd
import pytest

from churn_library import (CAT_COLUMNS, DATA_PATH, create_churn_flag,
                           encoder_helper, import_data, perform_eda,
                           perform_feature_engineering, train_models)

warnings.filterwarnings("ignore")


logger = logging.getLogger()


@pytest.fixture()
def path():
    """Return the data path."""
    return DATA_PATH


def test_import(path):
    """Test data import - this example is completed for you to assist\
    with the other test functions."""
    try:
        churn_data = import_data(path)
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert isinstance(churn_data, pd.DataFrame)
        assert churn_data.shape[0] > 0
        assert churn_data.shape[1] > 0
        pytest.churn_data = churn_data
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows\
             and columns"
        )
        raise err


def test_create_churn_flag():
    """Test the function that creates the churn flag."""
    try:
        pytest.churn_data = create_churn_flag(pytest.churn_data)
        logger.info("Testing create_churn_flag: SUCCESS")
    except Exception as err:
        logger.error("The churn flag can't be created!")
        raise err


def test_eda():
    """Test the function that perfoms the EDA."""
    try:
        perform_eda(pytest.churn_data)
        logger.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logger.error("The EDA can't be performed!")
        raise err


def test_encoder_helper():
    """Test the function that encodes categories."""
    try:
        encoded_df = encoder_helper(
            pytest.churn_data, CAT_COLUMNS, target="Churn")
        logger.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logger.error("Target encoding can't be performed!")
        raise err

    try:
        assert set(CAT_COLUMNS).issubset(encoded_df.columns)
        pytest.encoded_df = encoded_df
    except Exception as err:
        logger.error("Target encoding can't be performed!")
        raise err


def test_perform_feature_engineering():
    """Test the function that perform the features engineering."""
    try:
        (
            pytest.X_train,
            pytest.X_test,
            pytest.y_train,
            pytest.y_test,
        ) = perform_feature_engineering(pytest.churn_data, target="Churn")
        logger.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logger.error("Target encoding can't be performed!")
        raise err


def test_train_models():
    """Test the function that train models."""
    try:
        train_models(
            pytest.X_train,
            pytest.X_test,
            pytest.y_train,
            pytest.y_test)
        logger.info("Testing test_train_models: SUCCESS")
    except Exception as err:
        logger.error("Models can't be trained!")
        raise err
