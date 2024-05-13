// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// NodewiseRegression
List NodewiseRegression(Eigen::VectorXd y, Eigen::MatrixXd covariates, Eigen::MatrixXd interactions, NumericVector gmixPath, NumericVector sglmixPath, NumericVector lambdaPath, int nlambda, double lambdaFactor, int maxit, double tol, bool verbose);
RcppExport SEXP _ncagr_NodewiseRegression(SEXP ySEXP, SEXP covariatesSEXP, SEXP interactionsSEXP, SEXP gmixPathSEXP, SEXP sglmixPathSEXP, SEXP lambdaPathSEXP, SEXP nlambdaSEXP, SEXP lambdaFactorSEXP, SEXP maxitSEXP, SEXP tolSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type covariates(covariatesSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type interactions(interactionsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type gmixPath(gmixPathSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sglmixPath(sglmixPathSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type lambdaPath(lambdaPathSEXP);
    Rcpp::traits::input_parameter< int >::type nlambda(nlambdaSEXP);
    Rcpp::traits::input_parameter< double >::type lambdaFactor(lambdaFactorSEXP);
    Rcpp::traits::input_parameter< int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(NodewiseRegression(y, covariates, interactions, gmixPath, sglmixPath, lambdaPath, nlambda, lambdaFactor, maxit, tol, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ncagr_NodewiseRegression", (DL_FUNC) &_ncagr_NodewiseRegression, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_ncagr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
