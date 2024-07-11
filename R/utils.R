#' @return n x (q+1)(p-1) matrix representing interactions btw responses and covs
#' @noRd

intxmx <- function(responses, covariates) {
  q <- ncol(covariates)
  icovariates <- cbind(1, covariates)
  result <- lapply(seq_len(q + 1), \(j) {
    responses * icovariates[, j]
  })
  result <- Reduce(cbind, result)
  result
}

#' @return symmetrized version of matrix mx
#' result_ij = result_ji is nonzero iff both mx_ij and mx_ji are nonzero,
#' in which case we choose the smaller value in magnitude.
#' @noRd
symmetrize <- function(mx, rule = "and") {
  if (rule == "and") {
    result <- mx * (abs(mx) < t(abs(mx))) + t(mx) * (t(abs(mx)) < abs(mx))
  } else {
    result <- mx * (abs(mx) >= t(abs(mx))) + t(mx) * (t(abs(mx)) >= abs(mx))
  }
  return(result)
}
