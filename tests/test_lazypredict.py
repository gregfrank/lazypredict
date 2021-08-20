#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `lazypredict` package."""

import pytest

from click.testing import CliRunner

from lazypredict import cli
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "lazypredict.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output

def test_classification():
    data = load_breast_cancer()  # load_iris()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.3,
        random_state=23
    )

    classi = LazyClassifier(
        verbose=0,
        predictions=True
    )

    models_c, predictions_c = classi.fit(
        X_train,
        X_test,
        Y_train,
        Y_test,
    )


def test_regression():
    boston = load_boston()
    X1, Y1 = shuffle(boston.data, boston.target, random_state=13)
    # X1 = X1.astype(np.float32)
    # Y1 = Y1.astype(np.float32)

    # split the dataset
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(
        X1,
        Y1,
        test_size=0.3,
        random_state=23
    )

    regr = LazyRegressor(
        verbose=1,
        predictions=True
    )

    models_r, predictions_r = regr.fit(
        X1_train,
        X1_test,
        Y1_train,
        Y1_test,
    )

