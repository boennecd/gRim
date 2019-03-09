#include <RcppArmadillo.h>
#include <vector>
#include <algorithm>

double cpp_LL
  (const arma::mat&, const arma::mat&, const arma::uword, const arma::uword);

/* S      covariance matrix 
 * n      number of observations
 * K      starting value for concentration matrix
 * nvar   number of variables
 * glen   length of each cliques
 * gg     concatenated clique indices
 * iter   maximum number of iterations
 * eps    convergence threshold
 */

//[[Rcpp::export]]
Rcpp::List cpp_ggmfit
(const arma::mat &S, const unsigned int n, arma::mat K, 
 const unsigned int nvar, const arma::uvec &glen, const arma::uvec &gg, 
 const unsigned int iter, const double eps)
{
  /* setup vector with clique indices */
  std::size_t ngen = gg.size(); /* number of cliques */
  std::vector<arma::uvec> clique_indices;
  clique_indices.reserve(ngen);
  /* TODO: avoid O(nvar * ngen) storage? */
  std::vector<arma::uvec> residual_indices;
  residual_indices.reserve(ngen);
  {
    auto gg_i = gg.begin();
    for(auto n_members : glen){
      arma::uvec clique(gg_i, n_members), residual(nvar - n_members);
      std::sort(clique.begin(), clique.end());
      
      auto r_i = residual.begin(), c_i = clique.begin();
      for(unsigned int i = 0; i < nvar and r_i != residual.end(); ++i){
        if(c_i != clique.end() and i == *c_i){
          ++c_i;
          continue;
        }
        
        *(r_i++) = i;
        
      }
      
      clique_indices.emplace_back(std::move(clique));
      residual_indices.emplace_back(std::move(residual));
      gg_i += n_members;
    }
  }

  /* run iterative proportional scaling  */
  if(ngen == 1)
    Rcpp::stop("Not implemented with ngen == 0");
  unsigned int i = 0;
  for(; i < iter; ++i){
    Rcpp::checkUserInterrupt();
    const arma::mat old_K = K;
    auto resiaul_it = residual_indices.begin();
    
    for(auto clique : clique_indices){
      const arma::uvec &residuals = *(resiaul_it++);
      arma::mat K_res_cli = K(residuals, clique);
      
      /* TODO: one inversion can be done once */
      K(clique, clique) = S(clique, clique).i() + 
        K_res_cli.t() * arma::solve(K(residuals, residuals), K_res_cli);
    }
    
    /* TODO: change convergence criteria */
    if(arma::norm(K - old_K) / arma::norm(K) < eps)
      break;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("logL") = cpp_LL(S, K, n, nvar), /* TODO: compute */
    Rcpp::Named("K") = std::move(K),
    Rcpp::Named("iter") = i + 1L);
}

double cpp_LL
  (const arma::mat &S, const arma::mat &K, const arma::uword nobs, 
   const arma::uword nvar)
{
  double l_det, l_sign, trace = 0.;
  arma::log_det(l_det, l_sign, K);
  auto *k = K.begin();
  for(auto s : S)
    trace += s * *(k++);
  
  return - (double)nobs * nvar * log(2. * M_PI) / 2. + nobs / 2. * (l_det - trace);
}
