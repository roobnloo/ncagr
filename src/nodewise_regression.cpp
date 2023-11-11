#include <Rcpp.h>
#include <RcppEigen.h>
#include <limits>
using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::depends(RcppEigen)]]

// For easy debugging
/*
void pprint(VectorXd v)
{
    std::cout << v << std::endl;
}

void pprint(MatrixXd m, int c)
{
    std::cout << m.col(c) << std::endl;
}
*/

double softThreshold(double x, double lambda)
{
    double diff = std::abs(x) - lambda;
    if (diff < 0)
    {
        return 0;
    }
    int signx;
    if (std::abs(x) <= std::numeric_limits<double>::epsilon())
    {
        signx = 0;
    }
    else
    {
        signx = (double(0) < x) - (x < double(0));
    }
    return signx * diff;
}

VectorXd softThreshold(const VectorXd &x, double lambda)
{
    VectorXd thresh(x.rows());
    for (int i = 0; i < x.rows(); ++i)
    {
        thresh(i) = softThreshold(x(i), lambda);
    }
    return thresh;
}

void applyRidgeUpdate(
    VectorXd &v, VectorXd &residual, const MatrixXd &U, double regmean)
{
    if (U.cols() != v.rows())
    {
        stop("Dimension mismatch during ridge update step!");
    }

    int p = U.rows();
    residual += U * v;
    auto regdiag = VectorXd::Constant(p, 2 * regmean).asDiagonal();
    MatrixXd UtUreg = U.transpose() * U / residual.rows();
    UtUreg += regdiag;
    v = UtUreg.llt().solve(U.transpose() * residual) / residual.rows();
    residual -= U * v;
    return;
}

void applyL1Update(
    Ref<VectorXd> b, VectorXd &residual, const MatrixXd &X, double penalty)
{
    int n = X.rows();
    int p = b.rows();

    if (residual.rows() != n || X.cols() != p)
    {
        stop("Dimension mismatch during L1 update step!");
    }

    for (int i = 0; i < b.rows(); ++i)
    {
        auto xslice = X.col(i);
        residual += b(i) * xslice;
        double xscale = xslice.squaredNorm() / n;
        b(i) = softThreshold(residual.dot(xslice) / n, penalty) / xscale;
        residual -= b(i) * xslice;
    }
    return;
}

double objective(
    const VectorXd &residual, const VectorXd &mean_coef, const MatrixXd &beta,
    double regmean, double lambda, double asparse)
{
    double quad_loss = residual.squaredNorm() / (2 * residual.rows());
    // double mean_obj = mean_coef.squaredNorm();
    double mean_obj = mean_coef.lpNorm<1>();
    double lasso_obj = beta.col(0).lpNorm<1>();
    double group_lasso_obj = 0;
    for (int i = 1; i < beta.cols(); ++i)
    {
        lasso_obj += beta.col(i).lpNorm<1>();
        group_lasso_obj += beta.col(i).norm();
    }
    double result = quad_loss +
                    regmean * mean_obj +
                    asparse * lambda * lasso_obj +
                    (1 - asparse) * lambda * group_lasso_obj;
    return result;
}

double objective_sgl(
    const VectorXd &residual, const VectorXd &v, double lambda, double asparse)
{
    int n = residual.rows();
    return residual.squaredNorm() / (2 * n) +
           asparse * lambda * v.lpNorm<1>() +
           (1 - asparse) * lambda * v.norm();
}

// TODO: Is there a way to avoid returning a copy in this function?
VectorXd sglUpdateStep(
    const VectorXd &center, const VectorXd &grad,
    double step, double lambda, double asparse)
{
    VectorXd thresh = softThreshold(
        center - step * grad, step * asparse * lambda);
    double threshnorm = thresh.norm();
    if (threshnorm <= step * (1 - asparse) * lambda)
    {
        return VectorXd::Zero(center.rows());
    }
    double normterm = 1 - step * (1 - asparse) * lambda / threshnorm;
    if (normterm < 0)
    {
        return VectorXd::Zero(center.rows());
    }
    return normterm * thresh;
}

void applySparseGLUpdate(
    Ref<VectorXd> beta_grp, VectorXd &residual, const MatrixXd &intx,
    double lambda, double asparse, int maxit, double tol)
{
    int n = intx.rows();
    if (residual.rows() != n || intx.cols() != beta_grp.rows())
    {
        stop("Dimension mismatch during sparse group lasso update step!");
    }

    residual += intx * beta_grp;
    VectorXd threshold = softThreshold(
        intx.transpose() * residual / n, asparse * lambda);

    // If this subgradient condition holds, the entire group should be zero
    if (threshold.norm() <= (1 - asparse) * lambda)
    {
        beta_grp = VectorXd::Zero(beta_grp.rows());
        return;
    }

    double step_size = 1;
    VectorXd center_new(beta_grp);
    double obj = INFINITY;
    double objnew;
    for (int i = 0; i < maxit; ++i)
    {
        objnew = objective_sgl(
            residual - intx * beta_grp, beta_grp, lambda, asparse);
        if (std::abs(objnew - obj) < tol)
        {
            break;
        }
        if (i == maxit)
        {
            warning(
                "Inner loop within SGL descent exceeded max %i iterations",
                maxit);
        }
        obj = objnew;
        VectorXd center_old = center_new;
        VectorXd grp_fit = intx * beta_grp;
        VectorXd grad = -1 * intx.transpose() * (residual - grp_fit) / n;

        // Optimize the step size
        double lhs = INFINITY;
        double rhs = 0;
        double quad_loss_old = (residual - grp_fit).squaredNorm() / (2 * n);
        while (true)
        {
            center_new = sglUpdateStep(
                beta_grp, grad, step_size, lambda, asparse);
            VectorXd centerdiff = center_new - beta_grp;
            rhs = quad_loss_old + grad.dot(centerdiff) +
                  centerdiff.squaredNorm() / (2 * step_size);
            lhs = (residual - intx * center_new).squaredNorm() / (2 * n);
            if (lhs <= rhs)
                break;
            step_size *= 0.8;
        }

        // Nesterov momentum step
        beta_grp = center_old +
                      ((i + 1.0) / (i + 4)) * (center_new - center_old);
    }

    residual -= intx * beta_grp;
    return;
}

// TODO: is the variance correct?
double estimateVariance(
    const VectorXd &residual, const VectorXd &gamma, const MatrixXd &beta)
{
    int numNonZero = (beta.array().abs() > 0).count();
    // int nnZeroGroups = 0;
    // for (int i = 1; i < beta.cols(); ++i)
    // {
    //     if ((beta.col(i).array().abs() > 0).any())
    //     {
    //         ++nnZeroGroups;
    //     }
    // }
    // int nnZeroPop = (beta.col(0).array().abs() > 0).count();
    // double varhat = residual.squaredNorm() / (residual.rows() - 1 - nnZeroGroups - nnZeroPop);
    double varhat = residual.squaredNorm() / (residual.rows() - numNonZero);
    // std::cout << "varhat: " << varhat << std::endl;
    return varhat;
}

// TODO: This path of regmeans ignores the "beta" part of the residual.
NumericVector getRegMeanPath(int nregmean, const MatrixXd &covariates)
{
    if (nregmean <= 0)
    {
        stop("Number of mean penalty terms must be strictly positive!");
    }
    JacobiSVD<MatrixXd> svd(covariates);
    double largestSV = svd.singularValues()[0];

    NumericVector loglinInterp(nregmean);
    double regmeanFactor = 1e-3;
    double delta = log(regmeanFactor) / (nregmean - 1);
    for (int i = 0; i < nregmean; ++i)
    {
        loglinInterp[i] = largestSV * exp(i * delta);
    }
    return loglinInterp;
}

NumericVector getRegMeanPathSparse(
    int nregmean, const VectorXd &y, const MatrixXd &covariates,
    double regmeanFactor = 1e-4)
{
    if (nregmean <= 0)
    {
        stop("Number of mean penalty terms must be strictly positive!");
    }

    NumericVector loglinInterp(nregmean);
    double regmeanMax = (covariates.transpose() * y).lpNorm<Infinity>();
    if (nregmean == 1) {
        loglinInterp[0] = regmeanMax;
        return loglinInterp;
    }

    double delta = log(regmeanFactor) / (nregmean - 1);
    for (int i = 0; i < nregmean; ++i)
    {
        loglinInterp[i] = regmeanMax * exp(i * delta);
    }
    return loglinInterp;
}

// TODO: This path of lambdas ignores the "gamma" part of the residual.
NumericVector getLambdaPath(
    NumericVector inlambda, int nlambda, double lambdaFactor,
    const VectorXd &y, const std::vector<MatrixXd> &intxs)
{
    if (inlambda.size() > 0)
    {
        NumericVector inSorted = inlambda.sort(true);
        if (inSorted[inSorted.size() - 1] <= 0)
        {
            stop("Provided lambda terms are not all positive!");
        }
        return inSorted;
    }
    double lamdaMax = 0;
    for (MatrixXd intx : intxs)
    {
        double mc = (intx.transpose() * y).array().abs().maxCoeff();
        if (mc > lamdaMax)
        {
            lamdaMax = mc;
        }
    }

    NumericVector loglinInterp(nlambda);
    double delta = log(lambdaFactor) / (nlambda - 1);
    for (int i = 0; i < nlambda; ++i)
    {
        loglinInterp[i] = lamdaMax * exp(i * delta);
    }
    return loglinInterp;
}

struct RegressionResult {
    VectorXd resid;
    double varhat;
    double objval;
};

RegressionResult nodewiseRegressionInit(
    const VectorXd &y, const MatrixXd &response, const MatrixXd &covariates,
    const std::vector<MatrixXd> &intxs,
    VectorXd &gamma, MatrixXd &beta, // initial guess
    double lambda, double asparse, double regmean,
    int maxit, double tol, bool verbose)
{
    int p = response.cols() + 1;
    int q = covariates.cols();
    beta.resize(p-1, q+1);

    VectorXd residual = y - covariates * gamma - response * beta.col(0);
    for (int i = 0; i < (int) intxs.size(); ++i)
    {
        residual -= intxs[i] * beta.col(i + 1);
    }

    NumericVector objval(maxit + 1);
    objval[0] = objective(residual, gamma, beta, regmean, lambda, asparse);

    for (int i = 0; i < maxit; ++i)
    {
        // applyRidgeUpdate(gamma, residual, covariates, regmean);
        applyL1Update(gamma, residual, covariates, regmean);
        applyL1Update(beta.col(0), residual, response, lambda * asparse);

        for (int j = 0; j < q; ++j)
        {
            applySparseGLUpdate(
                beta.col(j + 1), residual, intxs[j],
                lambda, asparse, maxit, tol);
        }

        objval[i + 1] = objective(residual, gamma, beta, regmean, lambda, asparse);
        // if (verbose)
        //     std::cout << "Iteration: " << i << ":: obj:" << objval[i+1] << std::endl;

        if (i > 4 && std::abs(objval[i + 1] - objval[i - 1]) < 1e-20
            && std::abs(objval[i] - objval[i - 2]) < 1e-20)
        {
            // std::cout << "potential oscillation" << std::endl;
            stop("Potential oscillation!");
        }

        if (std::abs(objval[i + 1] - objval[i]) < tol)
        {
            objval = objval[Rcpp::Range(0, i + 1)];
            break;
        }
    }
    // if (verbose)
    //     std::cout << "Finished in " << objval.length() << " iterations" << std::endl;
    if (objval.length() == maxit + 1)
    {
        // std::cout << "Maximum iterations exceeded!" << std::endl;
        warning("Maximum iterations exceeded!");
    }

    beta.resize((p-1)*(q+1), 1);
    double varhat = estimateVariance(residual, gamma, beta);

    return RegressionResult {
        residual, varhat, objval[objval.size() - 1]
    };
}

// [[Rcpp::export]]
List NodewiseRegression(
    Eigen::VectorXd y, Eigen::MatrixXd response, Eigen::MatrixXd covariates, double asparse,
    NumericVector regmeanPath = NumericVector::create(), int nregmean = 10,
    NumericVector lambdaPath = NumericVector::create(),
    int nlambda = 100, double lambdaFactor = 1e-4,
    int maxit = 1000, double tol = 1e-8, bool verbose = true)
{
    int p = response.cols() + 1;
    int q = covariates.cols();

    if (response.rows() != covariates.rows() || y.rows() != response.rows())
    {
        stop("Responses and covariates must have the same number of observations!");
    }
    if (asparse < 0 || asparse > 1)
    {
        stop("Sparsity mixture parameter must be between zero and one!");
    }
    if (maxit <= 0 || tol <= 0)
    {
        stop("Maximium iteration and/or numerical tolerance are out of range!");
    }

    std::vector<MatrixXd> intxs(q);
    for (int i = 0; i < q; ++i)
    {
        MatrixXd intx =
            response.array().colwise() * covariates.col(i).array();
        intxs[i] = intx;
    }

    if (regmeanPath.size() == 0) // TODO: sort by increasing if nonempty
        regmeanPath = getRegMeanPathSparse(nregmean, y, covariates);
    nregmean = regmeanPath.size();

    lambdaPath = getLambdaPath(lambdaPath, nlambda, lambdaFactor, y, intxs);
    nlambda = lambdaPath.size(); // a bit of a hack for user-provided lambdas

    MatrixXd gammaFull(q, nlambda * nregmean);
    MatrixXd betaFull((p-1)*(q+1), nlambda * nregmean);
    MatrixXd varhatFull(nlambda, nregmean);
    MatrixXd residualFull(y.rows(), nlambda * nregmean);
    MatrixXd objectiveFull(nlambda, nregmean);

    for (int regmeanIdx = 0; regmeanIdx < nregmean; ++regmeanIdx)
    {
        RegressionResult regResult;
        MatrixXd beta(MatrixXd::Zero((p-1)*(q+1), 1));
        VectorXd gamma(VectorXd::Zero(q));
        for (int lambdaIdx = 0; lambdaIdx < nlambda; ++lambdaIdx)
        {
            // if (verbose)
                // std::cout << "Regression with lambda index " << lambdaIdx << std::endl;
            regResult = nodewiseRegressionInit(
                y, response, covariates, intxs, gamma, beta,
                lambdaPath[lambdaIdx], asparse, regmeanPath[regmeanIdx],
                maxit, tol, verbose);

            // Use gamma and beta as initializers for next lambda (warm-starts)
            int colIdx = nlambda * regmeanIdx + lambdaIdx;
            gammaFull.col(colIdx) = gamma;
            betaFull.col(colIdx) = beta;
            residualFull.col(colIdx) = regResult.resid;
            varhatFull(lambdaIdx, regmeanIdx) = regResult.varhat;
            objectiveFull(lambdaIdx, regmeanIdx) = regResult.objval;
        }
    }

    NumericVector gamma(wrap(gammaFull));
    gamma.attr("dim") = NumericVector::create(q, nlambda, nregmean);
    NumericVector beta(wrap(betaFull));
    beta.attr("dim") = NumericVector::create((p-1)*(q+1), nlambda, nregmean);
    NumericVector resid(wrap(residualFull));
    resid.attr("dim") = NumericVector::create(y.rows(), nlambda, nregmean);

    return List::create(
        Named("beta") = beta,
        Named("gamma") = gamma,
        Named("varhat") = varhatFull,
        Named("objval") = objectiveFull,
        Named("resid") = resid,
        Named("lambdas") = lambdaPath,
        Named("regmeans") = regmeanPath);
}
