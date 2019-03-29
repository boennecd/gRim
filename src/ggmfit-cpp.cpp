#include "arma.h"
#include <vector>
#include <algorithm>
#include <cmath> 

extern "C"
{
  #include "_utils_mat.h"
}

# define CHECK_LAPACK_INFO(info_var, meth_name)                           \
if(info_var != 0)                                                         \
  Rcpp::stop("'" #meth_name "' failed with INFO: " + std::to_string(info));

double cpp_LL
  (const arma::mat&, const arma::mat&, const arma::uword, const arma::uword);

Rcpp::List cpp_ggmfit_return
  (const arma::mat&, arma::mat&, const unsigned int, const unsigned int, 
   const unsigned int);

bool conv_criteria(const arma::mat&, const arma::mat&, const double);

static const char U_char = 'U';
static int one_int = 1L;

std::tuple<
  std::vector<arma::uvec>, std::vector<arma::uvec>, 
  size_t, size_t> get_clique_n_residual_indices
  (const unsigned int, const arma::uvec&, const arma::uvec&, const bool);

template<typename T>
arma::mat sym_mat_inv(T X){
  arma::mat Z = X; /* copy */
  int dim = Z.n_cols, info;
  
  dpotrf_wrap(&U_char, &dim, Z.memptr(), &dim, &info);
  CHECK_LAPACK_INFO(info, dpotrf)
  
  dpotri_wrap(&U_char, &dim, Z.memptr(), &dim, &info);
  CHECK_LAPACK_INFO(info, dpotri)
  
  Z = arma::symmatu(Z);
  
  return Z;
}

/* S      covariance matrix 
 * n      number of observations
 * K      starting value for concentration matrix
 * nvar   number of variables
 * glen   length of each cliques
 * gg     concatenated clique indices
 * iter   maximum number of iterations
 * eps    convergence threshold
 */

inline Rcpp::List return_quick
  (const arma::mat &S, const unsigned int n, arma::mat &K, 
   const unsigned int nvar){
  K = sym_mat_inv(S);
  return cpp_ggmfit_return(S, K, n, nvar, 0L);
}

//[[Rcpp::export]]
Rcpp::List cpp_ggmfit_reg
  (const arma::mat &S, const unsigned int n, arma::mat K, 
   const unsigned int nvar, const arma::uvec &glen, const arma::uvec &gg, 
   const unsigned int iter, const double eps, const unsigned int details)
{
  const std::size_t ngen = glen.size(); /* number of cliques */
  const arma::uword dim = K.n_cols;
  
  /* return quickly if possible */
  if(ngen == 1)
    return return_quick(S, n, K, nvar);
  
  /* make vector with indices of conditionally independent neighbors */
  std::vector<arma::uvec> non_zero_indices;
  unsigned int max_size = 0L;
  {
    std::vector<std::set<arma::uword> > unique_indices(dim);
    
    /* loop over cliques and add all neighbors. Could like be done way 
     * smarter... */
    auto gg_i = gg.begin();
    for(auto n_members : glen){
      if(n_members < 2L){
        gg_i += n_members;
        continue;
      }
      
      auto end = gg_i + n_members, start = gg_i;
      for(auto g1 = gg_i; g1 != end; ++g1)
        unique_indices[*g1].insert(start, end);
      
      gg_i += n_members;
    }
    
    /* remove own index */
    arma::uword c = 0L; 
    for(auto &u : unique_indices){
      auto idx = u.find(c++);
      if(idx != u.end())
        u.erase(idx);
    }
    
    /* make uvecs with indices */
    non_zero_indices.reserve(K.n_cols);
    for(auto u : unique_indices){
      non_zero_indices.emplace_back(u.size());
      auto *b = non_zero_indices.back().begin();
      for(auto new_val : u)
        *(b++) = new_val;
      
      if(u.size() > max_size)
        max_size = u.size();
    }
  }
  
  /* run regressions and repeat until convergence */
  std::unique_ptr<double[]> 
  W_sub_mem(new double[max_size * max_size]), beta_mem(new double[max_size]);
  arma::mat W = K.i();
  unsigned int i = 0L;
  bool has_conv = false;
  arma::vec w_new(dim);
  for(; i < iter; ++i){
    Rcpp::checkUserInterrupt();
    
    const arma::mat W_old = W;
    unsigned int j = 0L;
    for(auto idx = non_zero_indices.cbegin();
        idx != non_zero_indices.cend(); ++idx, ++j){
      unsigned int N_non_zero = idx->size();
      if(N_non_zero < 1L){
        for(arma::uword k = 0; k < dim; ++k){
          if(k == j)
            continue;
          W(j, k) = W(k, j) = 0.;
          
          if(has_conv)
            K(j, k) = K(k, j) = 0.;
        }
        
        continue;
        
      }
      
      arma::mat W_sub(W_sub_mem.get(), N_non_zero, N_non_zero, false);
      arma::vec beta(beta_mem.get(), N_non_zero, false);
      
      W_sub = W(*idx, *idx);
      beta = S.unsafe_col(j)(*idx);
      
      int n_res_i = N_non_zero, info;
      dposv_wrap(
        &U_char, &n_res_i, &one_int, W_sub.memptr(), 
        &n_res_i, beta.memptr(), &n_res_i, &info);
      CHECK_LAPACK_INFO(info, dposv);
      
      /* notice: we also include the j'th element but we iterate past it */
      w_new = W.cols(*idx) * beta;
      
      auto new_ele = w_new.begin();
      for(arma::uword k = 0; k < dim; ++k, ++new_ele){
        if(k == j)
          continue;
        W(j, k) = W(k, j) = *new_ele;
        
      }
      
      if(!has_conv)
        continue;
      
      const double denum_fac = 
        S(j, j) - arma::dot(W.unsafe_col(j)(*idx), beta);
      const arma::uword *idx_k = idx->cbegin(); /* they are sorted */
      const double *b = beta.cbegin();
      for(arma::uword k = 0; k < dim; ++k){
        if(k == j){
          K(j, j) = 1. / denum_fac;
          continue;
          
        }
        
        if(idx_k != idx->cend() and k == *idx_k){
          K(j, k) = K(k, j) = - *(b++) / denum_fac;
          ++idx_k;
          continue;
          
        }
        
        K(j, k) = K(k, j) = 0.;
      }
    }
    
    if(has_conv)
      break;
    
    /* take one more iteration where we set the concentration matrix */
    has_conv = conv_criteria(W, W_old, eps);
  }
  
  return cpp_ggmfit_return(S, K, n, nvar, i);
}

//[[Rcpp::export]]
Rcpp::List cpp_ggmfit
(const arma::mat &S, const unsigned int n, arma::mat K, 
 const unsigned int nvar, const arma::uvec &glen, const arma::uvec &gg, 
 const unsigned int iter, const double eps, const unsigned int details)
{
  const std::size_t ngen = glen.size(); /* number of cliques */
  
  /* return quickly if possible */
  if(ngen == 1)
    return return_quick(S, n, K, nvar);
  
  /* setup vector with clique indices */
  std::size_t max_clique_size, max_resid_size;
  std::vector<arma::uvec> clique_indices, residual_indices;
  {
    auto tmp = get_clique_n_residual_indices(nvar, glen, gg, true);
    clique_indices   = std::move(std::get<0>(tmp));
    residual_indices = std::move(std::get<1>(tmp));
    max_clique_size  = std::get<2>(tmp);
    max_resid_size   = std::get<3>(tmp);
  }
  
  /* allocate memory to use */
  std::unique_ptr<double[]> 
    K_res_res_mem(new double[max_resid_size * max_resid_size]),
    K_res_cli_mem(new double[max_resid_size * max_clique_size]);
  
  /* compute and store inverse covariance matrices for cliques. 
   * This can be done in parallel but it is not the bottleneck */
  std::vector<arma::mat> inv_covars;
  inv_covars.reserve(ngen);
  for(auto clique : clique_indices)
    inv_covars.emplace_back(sym_mat_inv(S(clique, clique)));
  
  /* run iterative proportional scaling */
  double prev_ll = 0., ll = 0.;
  if(details > 0L){
    ll = cpp_LL(S, K, n, nvar);
    Rprintf("Initial logL: %14.6f \n", ll);
  }
  unsigned int i = 0;
  arma::mat S_working = sym_mat_inv(K);
  for(; i < iter; ++i){
    Rcpp::checkUserInterrupt();
    
    const arma::mat old_K = K, S_working_old = std::move(S_working);
    prev_ll = ll;
    
    auto resiaul_it = residual_indices.begin();
    auto S_cli_cli_inv = inv_covars.begin();
    for(auto clique : clique_indices){
      const arma::uvec &residuals = *(resiaul_it++);
      size_t n_res = residuals.size();
      arma::mat 
        K_res_cli(K_res_cli_mem.get(), n_res, clique.size(), false), 
        K_res_res(K_res_res_mem.get(), n_res, n_res        , false);
      K_res_cli = K(residuals, clique);
      K_res_res = K(residuals, residuals);
      
      int n_res_i = n_res, nrhs = clique.size(), info;
      dposv_wrap(
        &U_char, &n_res_i, &nrhs, K_res_res.memptr(), 
        &n_res_i, K_res_cli.memptr(), &n_res_i, &info);
      CHECK_LAPACK_INFO(info, dposv)
      
      K(clique, clique) =  *(S_cli_cli_inv++) +
        K(clique, residuals) * K_res_cli;
    }
    
    S_working = sym_mat_inv(K);
    if(conv_criteria(S_working, S_working_old, eps))
      break;
    
    if(details > 0L){
      ll = cpp_LL(S, K, n, nvar);
      Rprintf("Iteration: %3i logL: %14.6f diff logL: %20.13f\n", 
              i + 1L, ll, ll - prev_ll);
    }
  }
  
  return cpp_ggmfit_return(S, K, n, nvar, i + 1L);
}

//[[Rcpp::export]]
Rcpp::List cpp_ggmfit_wood
  (const arma::mat &S, const unsigned int n, arma::mat K, 
   const unsigned int nvar, const arma::uvec &glen, const arma::uvec &gg, 
   const unsigned int iter, const double eps, const unsigned int details)
{
  const std::size_t ngen = glen.size(); /* number of cliques */
  
  /* return quickly if possible */
  if(ngen == 1)
    return return_quick(S, n, K, nvar);
  
  /* setup vector with clique indices */
  std::size_t max_clique_size;
  std::vector<arma::uvec> clique_indices;
  {
    auto tmp = get_clique_n_residual_indices(nvar, glen, gg, false);
    clique_indices  = std::move(std::get<0>(tmp));
    max_clique_size = std::move(std::get<2>(tmp));
  }
  
  /* allocate memory to use */
  std::unique_ptr<double[]> 
    S_clique_mem      (new double[max_clique_size * nvar]),
    Delta_S_clique_mem(new double[max_clique_size * nvar]);
  
  /* compute and store inverse covariance matrices for cliques. 
   * This can be done in parallel but it is not the bottleneck */
  std::vector<arma::mat> inv_covars;
  inv_covars.reserve(ngen);
  for(auto clique : clique_indices)
    inv_covars.emplace_back(sym_mat_inv(S(clique, clique)));
  
  /* run iterative proportional scaling */
  unsigned int i = 0;
  int nrhs = S.n_cols;
  std::unique_ptr<int[]> ipiv(new int[nrhs]);
  arma::mat S_working = sym_mat_inv(K);
  for(; i < iter; ++i){
    Rcpp::checkUserInterrupt();
    
    const arma::mat S_working_old = S_working;
    
    auto S_cli_cli_inv = inv_covars.begin();
    for(auto clique : clique_indices){
      /* Use Binomial inverse theorem as Delta may be singular
       * https://en.wikipedia.org/wiki/Woodbury_matrix_identity#Binomial_inverse_theorem */
      arma::mat S_clique(clique.size(), nvar);
      S_clique = S_working.rows(clique);
      arma::mat Delta = 
        *(S_cli_cli_inv++) - sym_mat_inv(S_working(clique, clique));
      
      arma::mat Delta_S_clique(clique.size(), nvar); 
      Delta_S_clique = Delta * S_clique;
      arma::mat H = Delta_S_clique.cols(clique);
      H.diag() += 1.;
      
      int dim = H.n_cols, info;
      dgesv_wrap(&dim, &nrhs, H.memptr(), &dim, ipiv.get(), 
                 Delta_S_clique.memptr(), &dim, &info);
      CHECK_LAPACK_INFO(info, dgesv)
      
      S_working -= S_clique.t() * Delta_S_clique;
      
    }
    
    if(conv_criteria(S_working, S_working_old, eps))
      break;
  }
  
  K = sym_mat_inv(S_working);
  return cpp_ggmfit_return(S, K, n, nvar, i + 1L);
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

bool conv_criteria(
    const arma::mat &new_mat, const arma::mat &old_mat, const double eps){
  const arma::uword dim = new_mat.n_cols; /* assume symmetrical */
  
  for(arma::uword i = 0; i < dim; ++i)
    for(arma::uword j = 0; j <= i; ++j){
      double num = std::abs(new_mat(i, j) - old_mat(i, j));
      double den = std::sqrt(
        new_mat(i, i) * new_mat(j, j) + new_mat(i, j) * new_mat(i, j));
      
      if(num/den > eps)
        return false;
    }
  
  return true;
}

std::tuple<
  std::vector<arma::uvec>, std::vector<arma::uvec>, 
  size_t, size_t> get_clique_n_residual_indices
  (const unsigned int nvar, const arma::uvec &glen, const arma::uvec &gg, 
   const bool with_residuals)
  {
    const std::size_t ngen = glen.size();
    std::size_t max_clique_size = 0L, max_resid_size = 0L;
    std::vector<arma::uvec> clique_indices;
    clique_indices.reserve(ngen);
    /* TODO: avoid O(nvar * ngen) storage and do a bit extra computation? */
    std::vector<arma::uvec> residual_indices;
    if(with_residuals)
      residual_indices.reserve(ngen);
    {
      auto gg_i = gg.begin();
      for(auto n_members : glen){
        arma::uvec clique(gg_i, n_members), residual(nvar - n_members);
        std::sort(clique.begin(), clique.end());
        
        if(with_residuals){
          auto r_i = residual.begin(), c_i = clique.begin();
          for(unsigned int i = 0; i < nvar and r_i != residual.end(); ++i){
            if(c_i != clique.end() and i == *c_i){
              ++c_i;
              continue;
            }
            
            *(r_i++) = i;
            
          }
        }
        
        if(clique.size() > max_clique_size)
          max_clique_size = clique.size();
        if(with_residuals and residual.size() > max_resid_size)
          max_resid_size = residual.size();
        
        clique_indices.emplace_back(std::move(clique));
        if(with_residuals)
          residual_indices.emplace_back(std::move(residual));
        gg_i += n_members;
      }
    }
    
    return std::make_tuple(
      std::move(clique_indices), std::move(residual_indices), 
      max_clique_size, max_resid_size);
  }

Rcpp::List cpp_ggmfit_return
  (const arma::mat &S, arma::mat &K, const unsigned int n, 
   const unsigned int nvar, const unsigned int iter)
{
  return Rcpp::List::create(
    Rcpp::Named("logL") = cpp_LL(S, K, n, nvar),
    Rcpp::Named("K") = std::move(K),
    Rcpp::Named("iter") = iter);
}

