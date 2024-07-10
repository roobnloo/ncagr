#' Run Natural Covariate-adjusted Graphical Regression
#'
#' @param responses \eqn{n \times p} matrix of responses
#' @param covariates \eqn{n \times q} matrix of covariates
#' @param gmixpath A path of ncagr mixing parameters. Default is 0.1, 0.2, ..., 0.9.
#' @param sglmixpath A path of sparse-group lasso mixing parameter with \eqn{0\leq \alpha \leq 1}. Default is 0.75.
#' @param nlambda The number of lambda values to use for cross-validation - default is 100.
#' @param lambdafactor The smallest value of lambda as a fraction of the maximum lambda.
#' @param maxit The maximum number of iterations. Default is \eqn{3\times 10^6}.
#' @param tol The convergence threshhold for optimization. Default is \eqn{10^{-6}}.
#' @param nfolds Number of folds for cross-validation. Default is 5.
#' @param verbose If TRUE, prints progress messages. Default is TRUE.
#' @param ncores Runs the nodewise regressions in parallel using that many cores. Default is 1.
#' @param adaptive Use adaptive weights when fitting nodewise regressions. Default is TRUE.
#' @useDynLib ncagr
#' @importFrom Rcpp sourceCpp
#' @importFrom abind abind
#' @importFrom Matrix colMeans colSums
#' @importFrom stats sd
#' @import parallel
#' @export
ncagr <- function(responses, covariates, gmixpath = seq(0, 0.9, by = 0.1),
                  sglmixpath = 0.75, nlambda = 100,
                  lambdafactor = 1e-8, maxit = 3e6, tol = 1e-6, nfolds = 5,
                  verbose = TRUE, ncores = 1, adaptive = TRUE) {
  stopifnot(
    is.matrix(responses), is.matrix(covariates),
    nrow(responses) == nrow(covariates),
    all(gmixpath >= 0), all(gmixpath <= 1),
    all(sglmixpath >= 0), all(sglmixpath <= 1)
  )

  p <- ncol(responses)
  q <- ncol(covariates)
  bveclength <- (p - 1) * (q + 1)

  # Initial run to get lambdas for each response
  ngmix <- length(gmixpath)
  # nsglmix <- length(sglmixpath)
  lambda <- matrix(nrow = nlambda, ncol = p)
  beta <- matrix(nrow = p, ncol = bveclength)
  gamma <- matrix(nrow = p, ncol = q)
  l1_weights <- matrix(nrow = q + bveclength, ncol = p)
  l2_weights <- matrix(nrow = q, ncol = p)
  cvm <- array(dim = c(nlambda, ngmix, p))
  sigma2 <- numeric(p)
  mse <- numeric(p)
  cv_lambda_idx <- numeric(p)
  cv_gmix_idx <- numeric(p)

  cov_sds <- apply(covariates, 2, stats::sd)
  cov_scale <- scale(covariates)
  intx <- intxmx(responses, covariates)
  intx_sds <- apply(intx, 2, sd)
  intx_scale <- scale(intx)

  nodewise <- function(node) {
    y <- responses[, node] - mean(responses[, node])
    intx_scale_node <- intx_scale[, -(seq(0, q) * p + node)]
    nodereg <- cv_ncagr_node(
      y, cbind(cov_scale, intx_scale_node), p, q, nlambda, lambdafactor, gmixpath, sglmixpath,
      maxit, tol, nfolds, adaptive
    )
    if (verbose) {
      message(node, " ", appendLF = FALSE)
    }

    return(list(
      gamma = nodereg$gamma / cov_sds,
      beta = nodereg$beta / intx_sds[-(seq(0, q) * p + node)],
      sigma2 = nodereg$sigma2,
      mse = nodereg$mse,
      lambda = nodereg$lambda,
      cvm = nodereg$cvm,
      cv_lambda_idx = nodereg$cv_lambda_idx,
      cv_gmix_idx = nodereg$cv_gmix_idx,
      wl1 = nodereg$wl1,
      wl2 = nodereg$wl2
    ))
  }

  message("Running nodewise regressions...")

  if (ncores > 1) {
    reg_result <- parallel::mclapply(seq_len(p), nodewise, mc.cores = ncores)
  } else {
    reg_result <- lapply(seq_len(p), nodewise)
  }

  for (node in seq_len(p)) {
    gamma[node, ] <- reg_result[[node]]$gamma
    beta[node, ] <- reg_result[[node]]$beta
    sigma2[node] <- reg_result[[node]]$sigma2
    mse[node] <- reg_result[[node]]$mse
    lambda[, node] <- reg_result[[node]]$lambda
    cvm[, , node] <- reg_result[[node]]$cvm
    cv_lambda_idx[node] <- reg_result[[node]]$cv_lambda_idx
    cv_gmix_idx[node] <- reg_result[[node]]$cv_gmix_idx
    l1_weights[, node] <- reg_result[[node]]$wl1
    l2_weights[, node] <- reg_result[[node]]$wl2
  }

  message("\nFinished regressions.")

  bhat_tens <- array(0, dim = c(p, p, q + 1))
  for (i in seq_len(p)) {
    bhat_tens[i, -i, ] <- -beta[i, ] # / sigma2[i]
  }

  # for (i in seq_len(q+1)) {
  #   bhat_tens[, , i] <- symmetrize_sparse(bhat_tens[, , i])
  # }

  for (i in seq_len(p)) {
    bhat_tens[i, -i, ] <- bhat_tens[i, -i, ] / sigma2[i]
  }

  bhat_symm <- abind::abind(
    apply(bhat_tens, 3, symmetrize, simplify = FALSE),
    along = 3
  )

  outlist <- list(
    gamma = gamma,
    beta = bhat_symm,
    beta_raw = bhat_tens,
    sigma2 = sigma2,
    mse = mse,
    lambda = lambda,
    sglmixpath = sglmixpath,
    gmixpath = gmixpath,
    cvm = cvm,
    cv_lambda_idx = cv_lambda_idx,
    # cv_sglmix_idx = cv_sglmix_idx,
    cv_gmix_idx = cv_gmix_idx,
    l1_weights = l1_weights,
    l2_weights = l2_weights
  )
  class(outlist) <- "ncagr"

  return(outlist)
}
