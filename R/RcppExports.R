# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

NodewiseRegression <- function(y, covariates, interactions, gmixPath, sglmixPath, wl1, wl2, lambdaPath = as.numeric( c()), nlambda = 100L, lambdaFactor = 1e-4, maxit = 1000L, tol = 1e-8, verbose = FALSE) {
    .Call('_ncagr_NodewiseRegression', PACKAGE = 'ncagr', y, covariates, interactions, gmixPath, sglmixPath, wl1, wl2, lambdaPath, nlambda, lambdaFactor, maxit, tol, verbose)
}

