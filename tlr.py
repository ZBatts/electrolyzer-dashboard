import numpy as np
from numpy.linalg import inv
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

class TransferLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lambda_, fit_intercept=True, penalty_matrix=None):
        """
        :param lambda_: Regularization parameter; higher values constrain the adjustment (gamma).
                        Setting lambda_ = 0 fits only the target data; high values limit gamma to 0.
        :param fit_intercept: If True, an intercept term will be included in the model.
        :param penalty_matrix: Optional, array-like of shape (n_features,); penalty matrix for regularization.
        """
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.penalty_matrix = penalty_matrix

    def fit(self, X, y, base_coefficients):
        """
        Fits the model, adapting the base coefficients using the provided data.
        
        :param X: array-like of shape (n_samples, n_features) - Training data.
        :param y: array-like of shape (n_samples,) - Target values.
        :param base_coefficients: array-like of shape (n_features,) - Initial coefficients from source data.
        :return: self - Fitted model instance.
        """
        
        # Add intercept column if needed
        if self.fit_intercept:
            Xs = np.concatenate([X, np.ones((len(X), 1))], axis=1)
            beta = np.append(base_coefficients, 0)  # Extend beta with intercept term
        else:
            Xs = X
            beta = base_coefficients
        
        # Set up penalty matrix
        if self.penalty_matrix is None:
            Xt = np.eye(Xs.shape[1])  # Identity matrix by default
        else:
            Xt = np.diag(self.penalty_matrix)  # Use provided penalty matrix

        # Compute residuals
        Y = y - Xs @ beta
        
        # Calculate gamma (adjustment term)
        self.gamma = inv(Xs.T @ Xs + self.lambda_ * Xt.T @ Xt) @ Xs.T @ Y
        
        # Adjust coefficients by adding gamma
        self.coef_ = beta + self.gamma
        return self

    def predict(self, X):
        """
        Predicts target values using the adapted coefficients.
        
        :param X: array-like of shape (n_samples, n_features) - Input data.
        :return: array of shape (n_samples,) - Predicted target values.
        """
        check_is_fitted(self, ["coef_"])
        if self.fit_intercept:
            X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        return X @ self.coef_


