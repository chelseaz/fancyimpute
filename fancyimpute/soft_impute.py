# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function, division

from six.moves import range
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_array

from .common import masked_mae, CompletionResult
from .solver import Solver

F32PREC = np.finfo(np.float32).eps


class SoftImpute(Solver):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.
    """
    def __init__(
            self,
            shrinkage_values=np.logspace(2, -2, 9),
            convergence_threshold=0.001,
            max_iters=100,
            max_rank=None,
            n_power_iterations=1,
            validation_frac=0.2,
            init_fill_method="zero",
            min_value=None,
            max_value=None,
            normalizer=None,
            verbose=True):
        """
        Parameters
        ----------
        shrinkage_value : list[float]
            Grid of values by which we shrink singular values on each iteration.

        convergence_threshold : float
            Minimum ration difference between iterations (as a fraction of
            the Frobenius norm of the current solution) before stopping.

        max_iters : int
            Maximum number of SVD iterations

        max_rank : int, optional
            Perform a truncated SVD on each iteration with this value as its
            rank.

        n_power_iterations : int
            Number of power iterations to perform with randomized SVD

        init_fill_method : str
            How to initialize missing values of data matrix, default is
            to fill them with zeros.

        min_value : float
            Smallest allowable value in the solution

        max_value : float
            Largest allowable value in the solution

        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods

        verbose : bool
            Print debugging info
        """
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value,
            normalizer=normalizer)
        self.shrinkage_values = shrinkage_values
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.n_power_iterations = n_power_iterations
        self.validation_frac = validation_frac
        self.verbose = verbose

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm = np.sqrt((old_missing_values ** 2).sum())
        # edge cases
        if old_norm == 0 or (old_norm < F32PREC and np.sqrt(ssd) > F32PREC):
            return False
        else:
            return (np.sqrt(ssd) / old_norm) < self.convergence_threshold

    def _svd_step(self, X, shrinkage_value, max_rank=None):
        """
        Returns reconstructed X from low-rank thresholded SVD and
        the rank achieved.
        """
        if max_rank:
            # if we have a max rank then perform the faster randomized SVD
            (U, s, V) = randomized_svd(
                X,
                max_rank,
                n_iter=self.n_power_iterations)
        else:
            # perform a full rank SVD using ARPACK
            (U, s, V) = np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)
        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        S_thresh = np.diag(s_thresh)
        return (rank, U_thresh, V_thresh, S_thresh)

    def get_validation_mask(self, X, observed_mask):
        validation_mask = np.random.random(size=X.shape) < self.validation_frac
        return np.logical_and(validation_mask, observed_mask)

    # Run SoftImpute for a set shrinkage_value
    def _solve(self, X_init, missing_mask, shrinkage_value, verbose=True):
        X_filled = X_init.copy()
        observed_mask = ~missing_mask

        for i in range(self.max_iters):
            rank, U_thresh, V_thresh, S_thresh = self._svd_step(
                X_filled,
                shrinkage_value,
                max_rank=self.max_rank)
            X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))
            X_reconstruction = self.clip(X_reconstruction)

            # print error on observed data
            if verbose:
                mae = masked_mae(
                    X_true=X_init,
                    X_pred=X_reconstruction,
                    mask=observed_mask)
                print(
                    "[SoftImpute] Iter %d: observed MAE=%0.6f rank=%d" % (
                        i + 1,
                        mae,
                        rank))

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstruction,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstruction[missing_mask]
            if converged:
                break
        if verbose:
            print("[SoftImpute] Stopped after iteration %d for lambda=%f" % (
                i + 1,
                shrinkage_value))

        return CompletionResult(
            X_filled=X_filled,
            U=U_thresh,
            V=V_thresh,
            S=S_thresh,
            rank=rank)

    def solve(self, X, missing_mask):
        X = check_array(X, force_all_finite=False)

        X_init = X

        observed_mask = ~missing_mask

        # move some observed entries into validation set for grid search
        # these will get added to missing mask
        # then clobber entries in missing mask and re-impute matrix
        # to ensure validation set is truly held out
        validation_mask = self.get_validation_mask(X, observed_mask)
        validation_missing_mask = np.logical_or(missing_mask, validation_mask)
        X_init_validation = X.copy()
        X_init_validation[validation_missing_mask] = np.nan
        self.fill(X_init_validation, validation_missing_mask, inplace=True)

        # perform grid search over shrinkage values
        # using warm starts as the paper recommends - way faster
        best_shrinkage_value = self.shrinkage_values[0]
        best_mae = np.inf
        best_completion_result = None
        completion_result = None

        for shrinkage_value in self.shrinkage_values:
            if self.verbose:
                print("Using lambda=%f" % shrinkage_value)

            if completion_result is None:
                X_warm_start = X_init_validation
            else:
                X_warm_start = completion_result.X_filled

            completion_result = self._solve(X_warm_start, 
                missing_mask=validation_missing_mask, 
                shrinkage_value=shrinkage_value, 
                verbose=False)

            # Use MAE on validation set as cross-validation metric; could try other metrics
            validation_mae = masked_mae(
                X_true=X_init,
                X_pred=completion_result.X_filled,
                mask=validation_mask)
            print("lambda=%f: validation MAE=%0.6f" % (shrinkage_value, validation_mae))

            # update best shrinkage value, if applicable
            if validation_mae < best_mae:
                print("lambda=%f has achieved best MAE so far" % shrinkage_value)
                best_shrinkage_value = shrinkage_value
                best_mae = validation_mae
                best_completion_result = completion_result

        # train model on original observed/missing split using lambda with best validation MAE
        return self._solve(X_init, 
            missing_mask=missing_mask, 
            shrinkage_value=best_shrinkage_value, 
            verbose=self.verbose)
