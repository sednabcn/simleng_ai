"""
Limited dependent variable and qualitative variables.

Includes binary outcomes, count data, (ordered) ordinal data and limited
dependent variables.

General References
--------------------

A.C. Cameron and P.K. Trivedi.  `Regression Analysis of Count Data`.
    Cambridge, 1998

G.S. Madalla. `Limited-Dependent and Qualitative Variables in Econometrics`.
    Cambridge, 1983.

W. Greene. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
"""
__all__ = ["Logit", "GenLogit"]

from statsmodels.compat.pandas import Appender

import warnings

import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom

from statsmodels import base
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationError,
    SpecificationWarning,
)
from ..data_manager.generation import Data_Generation

try:
    import cvxopt  # noqa:F401

    have_cvxopt = True
except ImportError:
    have_cvxopt = False


# TODO: When we eventually get user-settable precision, we need to change
#       this
FLOAT_EPS = np.finfo(float).eps

# Limit for exponentials to avoid overflow
EXP_UPPER_LIMIT = np.log(np.finfo(np.float64).max) - 1.0

# TODO: add options for the parameter covariance/variance
#       ie., OIM, EIM, and BHHH see Green 21.4

_discrete_models_shape_docs = """
"""

_discrete_results_docs = """
    %(one_line_description)s

    Parameters
    ----------
    model : A DiscreteModel instance
    params : array_like
        The parameters of a fitted model.
    hessian : array_like
        The hessian of the fitted model.
    scale : float
        A scale parameter for the covariance matrix.

    Attributes
    ----------
    df_resid : float
        See model definition.
    df_model : float
        See model definition.
    llf : float
        Value of the loglikelihood
    %(extra_attr)s"""

_l1_results_attr = """    nnz_params : int
        The number of nonzero parameters in the model.  Train with
        trim_params == True or else numerical error will distort this.
    trimmed : bool array
        trimmed[i] == True if the ith parameter was trimmed from the model."""

_get_start_params_null_docs = """
Compute one-step moment estimator for null (constant-only) model

This is a preliminary estimator used as start_params.

Returns
-------
params : ndarray
    parameter estimate based one one-step moment matching

"""

_check_rank_doc = """
    check_rank : bool
        Check exog rank to determine model degrees of freedom. Default is
        True. Setting to False reduces model initialization time when
        exog.shape[1] is large.
    """


# helper for MNLogit (will be generally useful later)
def _numpy_to_dummies(endog):
    if endog.ndim == 2 and endog.dtype.kind not in ["S", "O"]:
        endog_dummies = endog
        ynames = range(endog.shape[1])
    else:
        dummies = get_dummies(endog, drop_first=False)
        ynames = {i: dummies.columns[i] for i in range(dummies.shape[1])}
        endog_dummies = np.asarray(dummies, dtype=float)

        return endog_dummies, ynames

    return endog_dummies, ynames


def _pandas_to_dummies(endog):
    if endog.ndim == 2:
        if endog.shape[1] == 1:
            yname = endog.columns[0]
            endog_dummies = get_dummies(endog.iloc[:, 0])
        else:  # series
            yname = "y"
            endog_dummies = endog
    else:
        yname = endog.name
        endog_dummies = get_dummies(endog)
    ynames = endog_dummies.columns.tolist()

    return endog_dummies, ynames, yname


def _validate_l1_method(method):
    """
    As of 0.10.0, the supported values for `method` in `fit_regularized`
    are "l1" and "l1_cvxopt_cp".  If an invalid value is passed, raise
    with a helpful error message

    Parameters
    ----------
    method : str

    Raises
    ------
    ValueError
    """
    if method not in ["l1", "l1_cvxopt_cp"]:
        raise ValueError(
            "`method` = {method} is not supported, use either "
            '"l1" or "l1_cvxopt_cp"'.format(method=method)
        )


#### Private Model Classes ####


class DiscreteModelShape(base.LikelihoodModel):
    """
    Abstract class for discrete choice models.

    This class does not do anything itself but lays out the methods and
    call signature expected of child classes in addition to those of
    statsmodels.model.LikelihoodModel.
    """

    def __init__(self, endog, exog, check_rank=True, **kwargs):
        self._check_rank = check_rank
        super().__init__(endog, exog, **kwargs)
        self.raise_on_perfect_prediction = True

    def initialize(self):
        """
        Initialize is called by
        statsmodels.model.LikelihoodModel.__init__
        and should contain any preprocessing that needs to be done for a model.
        """
        if self._check_rank:
            # assumes constant
            rank = tools.matrix_rank(self.exog, method="qr")
        else:
            # If rank check is skipped, assume full
            rank = self.exog.shape[1]
        self.df_model = float(rank - 1)
        self.df_resid = float(self.exog.shape[0] - rank)

    def cdf(self, X):
        """
        The cumulative distribution function of the model.
        """
        raise NotImplementedError

    def pdf(self, X):
        """
        The probability density (mass) function of the model.
        """
        raise NotImplementedError

    def _check_perfect_pred(self, params, *args):
        endog = self.endog
        fittedvalues = self.cdf(np.dot(self.exog, params[: self.exog.shape[1]]))
        if self.raise_on_perfect_prediction and np.allclose(fittedvalues - endog, 0):
            msg = "Perfect separation detected, results not available"
            raise PerfectSeparationError(msg)

    @Appender(base.LikelihoodModel.fit.__doc__)
    def fit(
        self,
        start_params=None,
        method="newton",
        maxiter=35,
        full_output=1,
        disp=1,
        callback=None,
        **kwargs,
    ):
        """
        Fit the model using maximum likelihood.

        The rest of the docstring is from
        statsmodels.base.model.LikelihoodModel.fit
        """
        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass  # TODO: make a function factory to have multiple call-backs

        mlefit = super().fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            callback=callback,
            **kwargs,
        )

        return mlefit  # It is up to subclasses to wrap results

    def fit_regularized(
        self,
        start_params=None,
        method="l1",
        maxiter="defined_by_method",
        full_output=1,
        disp=True,
        callback=None,
        alpha=0,
        trim_mode="auto",
        auto_trim_tol=0.01,
        size_trim_tol=1e-4,
        qc_tol=0.03,
        qc_verbose=False,
        **kwargs,
    ):
        """
        Fit the model using a regularized maximum likelihood.

        The regularization method AND the solver used is determined by the
        argument method.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : 'l1' or 'l1_cvxopt_cp'
            See notes for details.
        maxiter : {int, 'defined_by_method'}
            Maximum number of iterations to perform.
            If 'defined_by_method', then use method defaults (see notes).
        full_output : bool
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool
            Set to True to print convergence messages.
        fargs : tuple
            Extra arguments passed to the likelihood function, i.e.,
            loglike(x,*args).
        callback : callable callback(xk)
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.
        alpha : non-negative scalar or numpy array (same size as parameters)
            The weight multiplying the l1 penalty term.
        trim_mode : 'auto, 'size', or 'off'
            If not 'off', trim (set to zero) parameters that would have been
            zero if the solver reached the theoretical minimum.
            If 'auto', trim params using the Theory above.
            If 'size', trim params if they have very small absolute value.
        size_trim_tol : float or 'auto' (default = 'auto')
            Tolerance used when trim_mode == 'size'.
        auto_trim_tol : float
            Tolerance used when trim_mode == 'auto'.
        qc_tol : float
            Print warning and do not allow auto trim when (ii) (above) is
            violated by this much.
        qc_verbose : bool
            If true, print out a full QC report upon failure.
        **kwargs
            Additional keyword arguments used when fitting the model.

        Returns
        -------
        Results
            A results instance.

        Notes
        -----
        Using 'l1_cvxopt_cp' requires the cvxopt module.

        Extra parameters are not penalized if alpha is given as a scalar.
        An example is the shape parameter in NegativeBinomial `nb1` and `nb2`.

        Optional arguments for the solvers (available in Results.mle_settings)::

            'l1'
                acc : float (default 1e-6)
                    Requested accuracy as used by slsqp
            'l1_cvxopt_cp'
                abstol : float
                    absolute accuracy (default: 1e-7).
                reltol : float
                    relative accuracy (default: 1e-6).
                feastol : float
                    tolerance for feasibility conditions (default: 1e-7).
                refinement : int
                    number of iterative refinement steps when solving KKT
                    equations (default: 1).

        Optimization methodology

        With :math:`L` the negative log likelihood, we solve the convex but
        non-smooth problem

        .. math:: \\min_\\beta L(\\beta) + \\sum_k\\alpha_k |\\beta_k|

        via the transformation to the smooth, convex, constrained problem
        in twice as many variables (adding the "added variables" :math:`u_k`)

        .. math:: \\min_{\\beta,u} L(\\beta) + \\sum_k\\alpha_k u_k,

        subject to

        .. math:: -u_k \\leq \\beta_k \\leq u_k.

        With :math:`\\partial_k L` the derivative of :math:`L` in the
        :math:`k^{th}` parameter direction, theory dictates that, at the
        minimum, exactly one of two conditions holds:

        (i) :math:`|\\partial_k L| = \\alpha_k`  and  :math:`\\beta_k \\neq 0`
        (ii) :math:`|\\partial_k L| \\leq \\alpha_k`  and  :math:`\\beta_k = 0`
        """
        _validate_l1_method(method)
        # Set attributes based on method
        cov_params_func = self.cov_params_func_l1

        ### Bundle up extra kwargs for the dictionary kwargs.  These are
        ### passed through super(...).fit() as kwargs and unpacked at
        ### appropriate times
        alpha = np.array(alpha)
        assert alpha.min() >= 0
        try:
            kwargs["alpha"] = alpha
        except TypeError:
            kwargs = dict(alpha=alpha)
        kwargs["alpha_rescaled"] = kwargs["alpha"] / float(self.endog.shape[0])
        kwargs["trim_mode"] = trim_mode
        kwargs["size_trim_tol"] = size_trim_tol
        kwargs["auto_trim_tol"] = auto_trim_tol
        kwargs["qc_tol"] = qc_tol
        kwargs["qc_verbose"] = qc_verbose

        ### Define default keyword arguments to be passed to super(...).fit()
        if maxiter == "defined_by_method":
            if method == "l1":
                maxiter = 1000
            elif method == "l1_cvxopt_cp":
                maxiter = 70

        ## Parameters to pass to super(...).fit()
        # For the 'extra' parameters, pass all that are available,
        # even if we know (at this point) we will only use one.
        extra_fit_funcs = {"l1": fit_l1_slsqp}
        if have_cvxopt and method == "l1_cvxopt_cp":
            from statsmodels.base.l1_cvxopt import fit_l1_cvxopt_cp

            extra_fit_funcs["l1_cvxopt_cp"] = fit_l1_cvxopt_cp
        elif method.lower() == "l1_cvxopt_cp":
            raise ValueError(
                "Cannot use l1_cvxopt_cp as cvxopt "
                "was not found (install it, or use method='l1' instead)"
            )

        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass  # make a function factory to have multiple call-backs

        mlefit = super().fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            callback=callback,
            extra_fit_funcs=extra_fit_funcs,
            cov_params_func=cov_params_func,
            **kwargs,
        )

        return mlefit  # up to subclasses to wrap results

    def cov_params_func_l1(self, likelihood_model, xopt, retvals):
        """
        Computes cov_params on a reduced parameter space
        corresponding to the nonzero parameters resulting from the
        l1 regularized fit.

        Returns a full cov_params matrix, with entries corresponding
        to zero'd values set to np.nan.
        """
        H = likelihood_model.hessian(xopt)
        trimmed = retvals["trimmed"]
        nz_idx = np.nonzero(~trimmed)[0]
        nnz_params = (~trimmed).sum()
        if nnz_params > 0:
            H_restricted = H[nz_idx[:, None], nz_idx]
            # Covariance estimate for the nonzero params
            H_restricted_inv = np.linalg.inv(-H_restricted)
        else:
            H_restricted_inv = np.zeros(0)

        cov_params = np.nan * np.ones(H.shape)
        cov_params[nz_idx[:, None], nz_idx] = H_restricted_inv

        return cov_params

    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.
        """
        raise NotImplementedError

    def _derivative_exog(self, params, exog=None, dummy_idx=None, count_idx=None):
        """
        This should implement the derivative of the non-linear function
        """
        raise NotImplementedError

    def _derivative_exog_helper(
        self, margeff, params, exog, dummy_idx, count_idx, transform
    ):
        """
        Helper for _derivative_exog to wrap results appropriately
        """
        from .discrete_margins import _get_count_effects, _get_dummy_effects

        if count_idx is not None:
            margeff = _get_count_effects(
                margeff, exog, count_idx, transform, self, params
            )
        if dummy_idx is not None:
            margeff = _get_dummy_effects(
                margeff, exog, dummy_idx, transform, self, params
            )

        return margeff


class BinaryModel(DiscreteModelShape):
    _continuous_ok = False

    def __init__(self, endog, exog, check_rank=True, **kwargs):
        # unconditional check, requires no extra kwargs added by subclasses
        self._check_kwargs(kwargs)
        super().__init__(endog, exog, check_rank, **kwargs)
        if not issubclass(self.__class__, MultinomialModel):
            if not np.all((self.endog >= 0) & (self.endog <= 1)):
                raise ValueError("endog must be in the unit interval.")

            if not self._continuous_ok and np.any(self.endog != np.round(self.endog)):
                raise ValueError("endog must be binary, either 0 or 1")

    def predict(self, params, exog=None, linear=False):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array_like
            Fitted parameters of the model.
        exog : array_like
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used.
        linear : bool, optional
            If True, returns the linear predictor dot(exog,params).  Else,
            returns the value of the cdf at the linear predictor.

        Returns
        -------
        array
            Fitted values at exog.
        """
        if exog is None:
            exog = self.exog
        if not linear:
            return self.cdf(np.dot(exog, params))
        else:
            return np.dot(exog, params)

    @Appender(DiscreteModelShape.fit_regularized.__doc__)
    def fit_regularized(
        self,
        start_params=None,
        method="l1",
        maxiter="defined_by_method",
        full_output=1,
        disp=1,
        callback=None,
        alpha=0,
        trim_mode="auto",
        auto_trim_tol=0.01,
        size_trim_tol=1e-4,
        qc_tol=0.03,
        **kwargs,
    ):
        _validate_l1_method(method)

        bnryfit = super().fit_regularized(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            callback=callback,
            alpha=alpha,
            trim_mode=trim_mode,
            auto_trim_tol=auto_trim_tol,
            size_trim_tol=size_trim_tol,
            qc_tol=qc_tol,
            **kwargs,
        )

        discretefit = L1BinaryResults(self, bnryfit)
        return L1BinaryResultsWrapper(discretefit)

    def _derivative_predict(self, params, exog=None, transform="dydx"):
        """
        For computing marginal effects standard errors.

        This is used only in the case of discrete and count regressors to
        get the variance-covariance of the marginal effects. It returns
        [d F / d params] where F is the predict.

        Transform can be 'dydx' or 'eydx'. Checking is done in margeff
        computations for appropriate transform.
        """
        if exog is None:
            exog = self.exog
        dF = self.pdf(np.dot(exog, params))[:, None] * exog
        if "ey" in transform:
            dF /= self.predict(params, exog)[:, None]
        return dF

    def _derivative_exog(
        self, params, exog=None, transform="dydx", dummy_idx=None, count_idx=None
    ):
        """
        For computing marginal effects returns dF(XB) / dX where F(.) is
        the predicted probabilities

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        # Note: this form should be appropriate for
        #   group 1 probit, logit, logistic, cloglog, heckprob, xtprobit
        if exog is None:
            exog = self.exog

        margeff = np.dot(self.pdf(np.dot(exog, params))[:, None], params[None, :])

        if "ex" in transform:
            margeff *= exog
        if "ey" in transform:
            margeff /= self.predict(params, exog)[:, None]

        return self._derivative_exog_helper(
            margeff, params, exog, dummy_idx, count_idx, transform
        )


class Logit(BinaryModel):
    __doc__ = """
    Logit Model

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    """ % {
        "params": base._model_params_doc,
        "extra_params": base._missing_param_doc + _check_rank_doc,
    }

    _continuous_ok = True

    def cdf(self, X):
        """
        The logistic cumulative distribution function

        Parameters
        ----------
        X : array_like
            `X` is the linear predictor of the logit model.  See notes.

        Returns
        -------
        1/(1 + exp(-X))

        Notes
        -----
        In the logit model,

        .. math:: \\Lambda\\left(x^{\\prime}\\beta\\right)=
                  \\text{Prob}\\left(Y=1|x\\right)=
                  \\frac{e^{x^{\\prime}\\beta}}{1+e^{x^{\\prime}\\beta}}
        """
        X = np.asarray(X)
        return 1 / (1 + np.exp(-X))

    def pdf(self, X):
        """
        The logistic probability density function

        Parameters
        ----------
        X : array_like
            `X` is the linear predictor of the logit model.  See notes.

        Returns
        -------
        pdf : ndarray
            The value of the Logit probability mass function, PMF, for each
            point of X. ``np.exp(-x)/(1+np.exp(-X))**2``

        Notes
        -----
        In the logit model,

        .. math:: \\lambda\\left(x^{\\prime}\\beta\\right)=\\frac{e^{-x^{\\prime}\\beta}}{\\left(1+e^{-x^{\\prime}\\beta}\\right)^{2}}
        """
        X = np.asarray(X)
        return np.exp(-X) / (1 + np.exp(-X)) ** 2

    def loglike(self, params):
        """
        Log-likelihood of logit model.

        Parameters
        ----------
        params : array_like
            The parameters of the logit model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----
        .. math::

           \\ln L=\\sum_{i}\\ln\\Lambda
           \\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        q = 2 * self.endog - 1
        X = self.exog
        return np.sum(np.log(self.cdf(q * np.dot(X, params))))

    def loglikeobs(self, params):
        """
        Log-likelihood of logit model for each observation.

        Parameters
        ----------
        params : array_like
            The parameters of the logit model.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----
        .. math::

           \\ln L=\\sum_{i}\\ln\\Lambda
           \\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        for observations :math:`i=1,...,n`

        where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        q = 2 * self.endog - 1
        X = self.exog
        return np.log(self.cdf(q * np.dot(X, params)))

    def score(self, params):
        """
        Logit model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\Lambda_{i}\\right)x_{i}
        """

        y = self.endog
        X = self.exog
        L = self.cdf(np.dot(X, params))
        return np.dot(y - L, X)

    def score_obs(self, params):
        """
        Logit model Jacobian of the log-likelihood for each observation

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        jac : array_like
            The derivative of the loglikelihood for each observation evaluated
            at `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\Lambda_{i}\\right)x_{i}

        for observations :math:`i=1,...,n`
        """

        y = self.endog
        X = self.exog
        L = self.cdf(np.dot(X, params))
        return (y - L)[:, None] * X

    def hessian(self, params):
        """
        Logit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i}\\Lambda_{i}\\left(1-\\Lambda_{i}\\right)x_{i}x_{i}^{\\prime}
        """
        X = self.exog
        L = self.cdf(np.dot(X, params))
        return -np.dot(L * (1 - L) * X.T, X)

    @Appender(DiscreteModelShape.fit.__doc__)
    def fit(
        self,
        start_params=None,
        method="newton",
        maxiter=35,
        full_output=1,
        disp=1,
        callback=None,
        **kwargs,
    ):
        bnryfit = super().fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            callback=callback,
            **kwargs,
        )

        discretefit = LogitResults(self, bnryfit)
        return BinaryResultsWrapper(discretefit)


class GenLogit(BinaryModel):
    __doc__ = """
    Logit Model

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    """ % {
        "params": base._model_params_doc,
        "extra_params": base._missing_param_doc + _check_rank_doc,
    }
    _continuous_ok = True

    def __init__(self, **kwargs):
        print("==========")
        print("I am here")
        print(exog.shape)
        for key, value in kwargs.items():
            print("value:", value)
        print(input("STOP"))

    def cdf(self, X):
        """
        The logistic cumulative distribution function

        Parameters
        ----------
        X : array_like
            `X` is the linear predictor of the logit model.  See notes.
        c: shape parameter
        Returns
        -------
        1/(1 + exp(-X))**c

        Notes
        -----
        In the genlogit model,

        .. math:: \\Lambda\\left(x^{\\prime}\\beta\\right)=
                  \\text{Prob}\\left(Y=1|x\\right)=
                 [ \\frac{e^{x^{\\prime}\\beta}}{1+e^{x^{\\prime}\\beta}}]^c
        """
        X = np.asarray(X)
        c = self.shape
        return 1 / (1 + np.exp(-X)) ** c

    def pdf(self, X):
        """
        The logistic probability density function

        Parameters
        ----------
        X : array_like
            `X` is the linear predictor of the logit model.  See notes.
        c : 'c' as a shape parameter
        Returns
        -------
        pdf : ndarray
            The value of the GenLogit probability mass function, PMF, for each
            point of X. ``c*np.exp(-x)/(1+np.exp(-X))**(c+1)``

        Notes
        -----
        In the logit model,

        .. math:: \\lambda\\left(x^{\\prime}\\beta\\right)=c*\\frac{e^{-x^{\\prime}\\beta}}{\\left(1+e^{-x^{\\prime}\\beta}\\right)^{c+1}}
        """
        X = np.asarray(X)
        c = self.shape
        return c * np.exp(-X) / (1 + np.exp(-X)) ** (c + 1)

    def loglike(self, params):
        """
        Log-likelihood of logit model.

        Parameters
        ----------
        params : array_like
            The parameters of the logit model.
        c : 'c' as a shape parameter
        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----
        .. math::

           \\ln L=\\sum_{i}\\ln\\Lambda
           \\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """

        X = self.exog
        y = self.endog
        Lcdf = self.cdf(np.dot(X, params))
        Lpdf = self.pdf(np.dot(X, params))
        return np.sum((2 * y - 1) * np.log(Lcdf) + (1 - y) * np.log(Lpdf))

    def loglikeobs(self, params):
        """
        Log-likelihood of logit model for each observation.

        Parameters
        ----------
        params : array_like
            The parameters of the logit model.
            c : 'c' as a shape parameter
        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----
        .. math::

           \\ln L=\\sum_{i}\\ln\\Lambda
           \\left(q_{i}x_{i}^{\\prime}\\beta\\right)

        for observations :math:`i=1,...,n`

        where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """

        X = self.exog
        y = self.endog
        Lcdf = self.cdf(np.dot(X, params))
        Lpdf = self.pdf(np.dot(X, params))
        return (2 * y - 1) * np.log(Lcdf) + (1 - y) * np.log(Lpdf)

    def score(self, params):
        """
        Logit model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model
        c : 'c' as a shape parameter
        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\Lambda_{i}\\right)x_{i}
        """
        X = self.exog
        y = self.endog
        Lcdf = self.cdf(np.dot(X, params))
        Lpdf = self.pdf(np.dot(X, params))
        Lgrad = (1 - y) * (1 - 2.0 * Lcdf) + (2.0 * y - 1.0) * (Lpdf / Lcdf)
        return np.dot(Lgrad, X)

    def jac(self, params):
        """
        GenLogit model Jacobian of the log-likelihood for each observation

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        jac : array_like
            The derivative of the loglikelihood for each observation evaluated
            at `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\Lambda_{i}\\right)x_{i}

        for observations :math:`i=1,...,n`
        """

        X = self.exog
        y = self.endog
        Lcdf = self.cdf(np.dot(X, params))
        Lpdf = self.pdf(np.dot(X, params))
        Lgrad = (1 - y) * (1 - 2.0 * Lcdf) + (2.0 * y - 1.0) * (Lpdf / Lcdf)
        return Lgrad[:, None] * X

    def hessian(self, params):
        """
        Logit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i}\\Lambda_{i}\\left(1-\\Lambda_{i}\\right)x_{i}x_{i}^{\\prime}
        """
        X = self.exog
        y = self.endog
        Lcdf = self.cdf(np.dot(X, params))
        Lpdf = self.pdf(np.dot(X, params))
        Lop = Lpdf / Lcdf
        Lgrad_grad = 2.0 * (1 - y) * Lpdf
        Lgrad_grad += (1 - 2.0 * y) * (1 - 2.0 * Lcdf) * Lop
        Lgrad_grad += (2.0 * y - 1.0) * (Lop**2)
        return -np.dot(Lgrad_grad * X.T, X)

    @Appender(DiscreteModelShape.fit.__doc__)
    def fit(
        self,
        start_params=None,
        method="newton",
        maxiter=35,
        full_output=1,
        disp=1,
        callback=None,
        **kwargs,
    ):
        bnryfit = super().fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            callback=callback,
            **kwargs,
        )

        discretefit = LogitResults(self, bnryfit)
        return BinaryResultsWrapper(discretefit)


### Results Class ###


class DiscreteResults(base.LikelihoodModelResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for the discrete dependent variable models.",
        "extra_attr": "",
    }

    def __init__(self, model, mlefit, cov_type="nonrobust", cov_kwds=None, use_t=None):
        # super(DiscreteResults, self).__init__(model, params,
        #        np.linalg.inv(-hessian), scale=1.)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self._cache = {}
        self.nobs = model.exog.shape[0]
        self.__dict__.update(mlefit.__dict__)

        if not hasattr(self, "cov_type"):
            # do this only if super, i.e. mlefit did not already add cov_type
            # robust covariance
            if use_t is not None:
                self.use_t = use_t
            if cov_type == "nonrobust":
                self.cov_type = "nonrobust"
                self.cov_kwds = {
                    "description": "Standard Errors assume that the "
                    + "covariance matrix of the errors is correctly "
                    + "specified."
                }
            else:
                if cov_kwds is None:
                    cov_kwds = {}
                from statsmodels.base.covtype import get_robustcov_results

                get_robustcov_results(
                    self, cov_type=cov_type, use_self=True, **cov_kwds
                )

    def __getstate__(self):
        # remove unpicklable methods
        mle_settings = getattr(self, "mle_settings", None)
        if mle_settings is not None:
            if "callback" in mle_settings:
                mle_settings["callback"] = None
            if "cov_params_func" in mle_settings:
                mle_settings["cov_params_func"] = None
        return self.__dict__

    @cache_readonly
    def prsquared(self):
        """
        McFadden's pseudo-R-squared. `1 - (llf / llnull)`
        """
        return 1 - self.llf / self.llnull

    @cache_readonly
    def llr(self):
        """
        Likelihood ratio chi-squared statistic; `-2*(llnull - llf)`
        """
        return -2 * (self.llnull - self.llf)

    @cache_readonly
    def llr_pvalue(self):
        """
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
        """
        return stats.distributions.chi2.sf(self.llr, self.df_model)

    def set_null_options(self, llnull=None, attach_results=True, **kwargs):
        """
        Set the fit options for the Null (constant-only) model.

        This resets the cache for related attributes which is potentially
        fragile. This only sets the option, the null model is estimated
        when llnull is accessed, if llnull is not yet in cache.

        Parameters
        ----------
        llnull : {None, float}
            If llnull is not None, then the value will be directly assigned to
            the cached attribute "llnull".
        attach_results : bool
            Sets an internal flag whether the results instance of the null
            model should be attached. By default without calling this method,
            thenull model results are not attached and only the loglikelihood
            value llnull is stored.
        **kwargs
            Additional keyword arguments used as fit keyword arguments for the
            null model. The override and model default values.

        Notes
        -----
        Modifies attributes of this instance, and so has no return.
        """
        # reset cache, note we need to add here anything that depends on
        # llnullor the null model. If something is missing, then the attribute
        # might be incorrect.
        self._cache.pop("llnull", None)
        self._cache.pop("llr", None)
        self._cache.pop("llr_pvalue", None)
        self._cache.pop("prsquared", None)
        if hasattr(self, "res_null"):
            del self.res_null

        if llnull is not None:
            self._cache["llnull"] = llnull
        self._attach_nullmodel = attach_results
        self._optim_kwds_null = kwargs

    @cache_readonly
    def llnull(self):
        """
        Value of the constant-only loglikelihood
        """
        model = self.model
        kwds = model._get_init_kwds().copy()
        for key in getattr(model, "_null_drop_keys", []):
            del kwds[key]
        # TODO: what parameters to pass to fit?
        mod_null = model.__class__(model.endog, np.ones(self.nobs), **kwds)
        # TODO: consider catching and warning on convergence failure?
        # in the meantime, try hard to converge. see
        # TestPoissonConstrained1a.test_smoke

        optim_kwds = getattr(self, "_optim_kwds_null", {}).copy()

        if "start_params" in optim_kwds:
            # user provided
            sp_null = optim_kwds.pop("start_params")
        elif hasattr(model, "_get_start_params_null"):
            # get moment estimates if available
            sp_null = model._get_start_params_null()
        else:
            sp_null = None

        opt_kwds = dict(method="bfgs", warn_convergence=False, maxiter=10000, disp=0)
        opt_kwds.update(optim_kwds)

        if optim_kwds:
            res_null = mod_null.fit(start_params=sp_null, **opt_kwds)
        else:
            # this should be a reasonably method case across versions
            res_null = mod_null.fit(
                start_params=sp_null,
                method="nm",
                warn_convergence=False,
                maxiter=10000,
                disp=0,
            )
            res_null = mod_null.fit(
                start_params=res_null.params,
                method="bfgs",
                warn_convergence=False,
                maxiter=10000,
                disp=0,
            )

        if getattr(self, "_attach_nullmodel", False) is not False:
            self.res_null = res_null

        return res_null.llf

    @cache_readonly
    def fittedvalues(self):
        """
        Linear predictor XB.
        """
        return np.dot(self.model.exog, self.params[: self.model.exog.shape[1]])

    @cache_readonly
    def resid_response(self):
        """
        Respnose residuals. The response residuals are defined as
        `endog - fittedvalues`
        """
        return self.model.endog - self.predict()

    @cache_readonly
    def aic(self):
        """
        Akaike information criterion.  `-2*(llf - p)` where `p` is the number
        of regressors including the intercept.
        """
        return -2 * (self.llf - (self.df_model + 1))

    @cache_readonly
    def bic(self):
        """
        Bayesian information criterion. `-2*llf + ln(nobs)*p` where `p` is the
        number of regressors including the intercept.
        """
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + 1)

    def _get_endog_name(self, yname, yname_list):
        if yname is None:
            yname = self.model.endog_names
        if yname_list is None:
            yname_list = self.model.endog_names
        return yname, yname_list

    def get_margeff(
        self, at="overall", method="dydx", atexog=None, dummy=False, count=False
    ):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available from the returned object.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)
            - 'eydx' - estimate semi-elasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables. For interpretations of these methods
            see notes below.
        atexog : array_like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        DiscreteMargins : marginal effects instance
            Returns an object that holds the marginal effects, standard
            errors, confidence intervals, etc. See
            `statsmodels.discrete.discrete_margins.DiscreteMargins` for more
            information.

        Notes
        -----
        Interpretations of methods:

        - 'dydx' - change in `endog` for a change in `exog`.
        - 'eyex' - proportional change in `endog` for a proportional change
          in `exog`.
        - 'dyex' - change in `endog` for a proportional change in `exog`.
        - 'eydx' - proportional change in `endog` for a change in `exog`.

        When using after Poisson, returns the expected number of events per
        period, assuming that the model is loglinear.
        """
        from statsmodels.discrete.discrete_margins import DiscreteMargins

        return DiscreteMargins(self, (at, method, atexog, dummy, count))

    def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
        """
        Summarize the Regression Results.

        Parameters
        ----------
        yname : str, optional
            The name of the endog variable in the tables. The default is `y`.
        xname : list[str], optional
            The names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model.
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title.
        alpha : float
            The significance level for the confidence intervals.

        Returns
        -------
        Summary
            Class that holds the summary tables and text, which can be printed
            or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : Class that hold summary results.
        """

        top_left = [
            ("Dep. Variable:", None),
            ("Model:", [self.model.__class__.__name__]),
            ("Method:", ["MLE"]),
            ("Date:", None),
            ("Time:", None),
            ("converged:", ["%s" % self.mle_retvals["converged"]]),
        ]

        top_right = [
            ("No. Observations:", None),
            ("Df Residuals:", None),
            ("Df Model:", None),
            ("Pseudo R-squ.:", ["%#6.4g" % self.prsquared]),
            ("Log-Likelihood:", None),
            ("LL-Null:", ["%#8.5g" % self.llnull]),
            ("LLR p-value:", ["%#6.4g" % self.llr_pvalue]),
        ]

        if hasattr(self, "cov_type"):
            top_left.append(("Covariance Type:", [self.cov_type]))

        if title is None:
            title = self.model.__class__.__name__ + " " + "Regression Results"

        # boiler plate
        from statsmodels.iolib.summary import Summary

        smry = Summary()
        yname, yname_list = self._get_endog_name(yname, yname_list)

        # for top of table
        smry.add_table_2cols(
            self,
            gleft=top_left,
            gright=top_right,
            yname=yname,
            xname=xname,
            title=title,
        )

        # for parameters, etc
        smry.add_table_params(
            self, yname=yname_list, xname=xname, alpha=alpha, use_t=self.use_t
        )

        if hasattr(self, "constraints"):
            smry.add_extra_txt(
                ["Model has been estimated subject to linear " "equality constraints."]
            )

        return smry

    def summary2(
        self, yname=None, xname=None, title=None, alpha=0.05, float_format="%.4f"
    ):
        """
        Experimental function to summarize regression results.

        Parameters
        ----------
        yname : str
            Name of the dependent variable (optional).
        xname : list[str], optional
            List of strings of length equal to the number of parameters
            Names of the independent variables (optional).
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title.
        alpha : float
            The significance level for the confidence intervals.
        float_format : str
            The print format for floats in parameters summary.

        Returns
        -------
        Summary
            Instance that contains the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : Class that holds summary results.
        """
        from statsmodels.iolib import summary2

        smry = summary2.Summary()
        smry.add_base(
            results=self,
            alpha=alpha,
            float_format=float_format,
            xname=xname,
            yname=yname,
            title=title,
        )

        if hasattr(self, "constraints"):
            smry.add_text(
                "Model has been estimated subject to linear " "equality constraints."
            )

        return smry


class OrderedResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for ordered discrete data.",
        "extra_attr": "",
    }
    pass


class BinaryResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for binary data",
        "extra_attr": "",
    }

    def pred_table(self, threshold=0.5):
        """
        Prediction table

        Parameters
        ----------
        threshold : scalar
            Number between 0 and 1. Threshold above which a prediction is
            considered 1 and below which a prediction is considered 0.

        Notes
        -----
        pred_table[i,j] refers to the number of times "i" was observed and
        the model predicted "j". Correct predictions are along the diagonal.
        """
        model = self.model
        actual = model.endog
        pred = np.array(self.predict() > threshold, dtype=float)
        bins = np.array([0, 0.5, 1])
        return np.histogram2d(actual, pred, bins=bins)[0]

    @Appender(DiscreteResults.summary.__doc__)
    def summary(self, yname=None, xname=None, title=None, alpha=0.05, yname_list=None):
        smry = super(BinaryResults, self).summary(
            yname, xname, title, alpha, yname_list
        )
        fittedvalues = self.model.cdf(self.fittedvalues)
        absprederror = np.abs(self.model.endog - fittedvalues)
        predclose_sum = (absprederror < 1e-4).sum()
        predclose_frac = predclose_sum / len(fittedvalues)

        # add warnings/notes
        etext = []
        if predclose_sum == len(fittedvalues):  # TODO: nobs?
            wstr = "Complete Separation: The results show that there is"
            wstr += "complete separation.\n"
            wstr += "In this case the Maximum Likelihood Estimator does "
            wstr += "not exist and the parameters\n"
            wstr += "are not identified."
            etext.append(wstr)
        elif predclose_frac > 0.1:  # TODO: get better diagnosis
            wstr = "Possibly complete quasi-separation: A fraction "
            wstr += "%4.2f of observations can be\n" % predclose_frac
            wstr += "perfectly predicted. This might indicate that there "
            wstr += "is complete\nquasi-separation. In this case some "
            wstr += "parameters will not be identified."
            etext.append(wstr)
        if etext:
            smry.add_extra_txt(etext)
        return smry

    @cache_readonly
    def resid_dev(self):
        """
        Deviance residuals

        Notes
        -----
        Deviance residuals are defined

        .. math:: d_j = \\pm\\left(2\\left[Y_j\\ln\\left(\\frac{Y_j}{M_jp_j}\\right) + (M_j - Y_j\\ln\\left(\\frac{M_j-Y_j}{M_j(1-p_j)} \\right) \\right] \\right)^{1/2}

        where

        :math:`p_j = cdf(X\\beta)` and :math:`M_j` is the total number of
        observations sharing the covariate pattern :math:`j`.

        For now :math:`M_j` is always set to 1.
        """
        # These are the deviance residuals
        # model = self.model
        endog = self.model.endog
        # exog = model.exog
        # M = # of individuals that share a covariate pattern
        # so M[i] = 2 for i = two share a covariate pattern
        M = 1
        p = self.predict()
        # Y_0 = np.where(exog == 0)
        # Y_M = np.where(exog == M)
        # NOTE: Common covariate patterns are not yet handled
        res = -(1 - endog) * np.sqrt(2 * M * np.abs(np.log(1 - p))) + endog * np.sqrt(
            2 * M * np.abs(np.log(p))
        )
        return res

    @cache_readonly
    def resid_pearson(self):
        """
        Pearson residuals

        Notes
        -----
        Pearson residuals are defined to be

        .. math:: r_j = \\frac{(y - M_jp_j)}{\\sqrt{M_jp_j(1-p_j)}}

        where :math:`p_j=cdf(X\\beta)` and :math:`M_j` is the total number of
        observations sharing the covariate pattern :math:`j`.

        For now :math:`M_j` is always set to 1.
        """
        # Pearson residuals
        # model = self.model
        endog = self.model.endog
        # exog = model.exog
        # M = # of individuals that share a covariate pattern
        # so M[i] = 2 for i = two share a covariate pattern
        # use unique row pattern?
        M = 1
        p = self.predict()
        return (endog - M * p) / np.sqrt(M * p * (1 - p))

    @cache_readonly
    def resid_response(self):
        """
        The response residuals

        Notes
        -----
        Response residuals are defined to be

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`.
        """
        return self.model.endog - self.predict()


class L1BinaryResults(BinaryResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "Results instance for binary data fit by l1 regularization",
        "extra_attr": _l1_results_attr,
    }

    def __init__(self, model, bnryfit):
        super(L1BinaryResults, self).__init__(model, bnryfit)
        # self.trimmed is a boolean array with T/F telling whether or not that
        # entry in params has been set zero'd out.
        self.trimmed = bnryfit.mle_retvals["trimmed"]
        self.nnz_params = (~self.trimmed).sum()
        self.df_model = self.nnz_params - 1
        self.df_resid = float(self.model.endog.shape[0] - self.nnz_params)


class LogitResults(BinaryResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Logit Model",
        "extra_attr": "",
    }

    @cache_readonly
    def resid_generalized(self):
        """
        Generalized residuals

        Notes
        -----
        The generalized residuals for the Logit model are defined

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`. This is the same as the `resid_response`
        for the Logit model.
        """
        # Generalized residuals
        return self.model.endog - self.predict()


class BinaryResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {
        "resid_dev": "rows",
        "resid_generalized": "rows",
        "resid_pearson": "rows",
        "resid_response": "rows",
    }
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs, _attrs)


wrap.populate_wrapper(BinaryResultsWrapper, BinaryResults)


class L1BinaryResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(L1BinaryResultsWrapper, L1BinaryResults)
