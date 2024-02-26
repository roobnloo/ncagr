#' Extract estimated coefficients from a 'ncagr' object
#'
#' @param fit An object of class [ncagr].
#' @return A list with components 'beta', an array of precision matrix
#'  components, and 'gamma', the estimate mean component.
#' @method coef ncagr
#' @export
coef.ncagr <- function(fit) {
  result <- list()
  result$beta <- fit$beta
  result$gamma <- fit$gamma
  return(result)
}

#' Predict from a 'ncagr' object
#'
#' This function predicts the mean and precision matrix for a new vector
#'  of covariates.
#' @param fit An object of class [ncagr].
#' @param newcovar A vector of covariates.
#' @return A list with components 'precision', the predicted precision matrix,
#'  and 'mean', the predicted mean given the new covariates.
#' @method predict ncagr
#' @export
predict.ncagr <- function(fit, newcovar) {
  q <- dim(fit$gamma)[2]
  dim(newcovar) <- NULL
  if (length(newcovar) != q)
    stop("Expected covariate vector of length ", q, ".")
  omega <- apply(fit$beta, c(1, 2), \(b) b %*% c(1, newcovar))
  diag(omega) <- 1 / fit$sigma2

  mu <- solve(omega, diag(diag(omega)) %*% fit$gamma %*% newcovar)
  return(list(precision = omega, mean = mu))
}
