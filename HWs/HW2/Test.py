# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:41:06 2021

@author: Izel Yazici 
"""

import unittest
import numpy as np
import pandas as pd
from LinearRegression import LinearReg
import numpy.testing as npt


class LinearRegressionTest(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame( [1.0, np.nan, 3.0, 1.2,1, np.nan,5])
        self.y = pd.DataFrame([10.0, np.nan, 20.0, 40.0, 20.0,np.nan,50])
        np.seterr(divide='ignore')
        beta, std_error, t1, t2, self.X_clean, self.y_clean = LinearReg(self.X, self.y)

    def test_nan(self):
        X = self.X.dropna().to_numpy()
        y = self.y.dropna().to_numpy()
        npt.assert_array_equal(X, self.X_clean)
        npt.assert_array_equal(y, self.y_clean)

    def test_empty(self):
        X = self.X.dropna()
        y = self.y.dropna()
        shape_X = np.shape(X)
        shape_y = np.shape(y)
        control_X = np.zeros(shape_X, dtype=bool)
        control_y = np.zeros(shape_y, dtype=bool)
        npt.assert_array_equal(control_X, pd.isnull(self.X_clean))
        npt.assert_array_equal(control_y, pd.isnull(self.y_clean))


if __name__ == "__main__":
    unittest.main()

