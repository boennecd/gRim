.rhsf2char <- function(f) {
  unlist(rhsf2list(f))
}

.list2pairs <- function(x){
  if (length(x)>0){
    if (!inherits(x,"list"))
      x <- list(x)
    x <- lapply(x, function(zzz) lapply(combn(zzz,2, simplify=FALSE), sort))
    x <- unlist(x, recursive=FALSE)
  }
  x
}

.toString <- function(x, col=' '){
  paste(x, collapse=col)
}

## Remove e and all higher order terms containing e from gen
##
## reduceSet(c(1,2,3,4),c(1,2))
## reduceSet(c(1,2,3,4),c(1,2,3))
.reduceSet <- function(gen,e){
  if (length(e)==2){
    i <- match(e,gen)
    ans <- list(gen[-i[1]], gen[-i[2]])
  } else {
    le <- length(e)
    lx <- length(gen)
    x2 <- unlist(lapply(le:(lx-1), function(l) combn(gen,l, simplify=FALSE)),
                 recursive=FALSE)

    i <- isin(x2, e, TRUE)
    ans <- x2[i==0]
  }
  ans
}

## Remove term e from each element of glist
##
## delete.term(list(c(1,2,3,4),c(2,3,4,5)),c(1,2))
## delete.term(list(c(1,2,3,4),c(2,3,4,5)),c(1,2,3))

.delete.term <-  function(glist, e){
  idx<-isin(glist,e,TRUE)
  zzz <- unlist(
                lapply(1:length(glist), function(i){
                  if (idx[i]==1)
                    .reduceSet(glist[[i]],e)
                  else
                    glist[i]
                }), recursive=FALSE)
  ans <- removeRedundant(zzz)
  ans
}

## Add e interaction to x
##
.add.term <- function(glist,e){
  removeRedundant(c(glist,list(e)))
}


## Sufficient statistics for saturate model in the form
## (
##    n.obs  = {n(i);  i \in I};
##    center = {mu(i); i \in I};
##    cov    = {S(i);  i \in I}
## )

### Known issues: The function should check that SS is of the right form
### (homogeneous and simplify)

.toString <- function(x, col=' '){
  paste(x, collapse=col)
}

print.MIparms <- function(x,simplify=TRUE,useN=FALSE, ...){

  cat(sprintf("MIparms: form=%s\n", class(x)[1]))

##   cat(sprintf("MIparms: form=%s, gentype=%s\n", class(x)[1], x$gentype))
##   aaa<-unlist(lapply(x, is.null))
##   cat("MIparms:",class(x)[1], "gentype:", x$gentype, "slots:", names(aaa)[!aaa],"\n")

  xx <- x
  if (useN)
    xx[[1]] <- xx[[1]]*xx$N

  if (x$gentype=="discrete"){
    if (simplify){
      print(as.numeric(xx[[1]]))
    } else {
      print(xx[[1]])
    }
  } else {
    if (simplify){
      print(rbind(
                  c(as.numeric(xx[[1]]),
                    rep(NA, ncol(xx[[3]]))),
                  cbind(xx[[2]],xx[[3]]))
            )
    } else {
      print(xx[1:3])
    }
  }
  cat("\n")
  return(invisible(x))
 }

.as.matrix <- .MIparms2matrix <- function(x,...){

  if (x$gentype=="discrete"){
    matrix(as.numeric(x[[1]]),nrow=1)
  } else {
    rbind(
          c(as.numeric(x[[1]]), rep(NA, ncol(x[[3]]))),
          cbind(x[[2]],x[[3]]))
  }
}

.logdet <- function(x){
  z <- determinant(x, logarithm = TRUE)
  c(z$sign * z$modulus)
}

.pd.det <- function(x,dx=dim(x)[1]){
  y <- c(chol(x))[1L + 0L:(dx[1L] - 1L) * (dx[1L] + 1L)]
  prod(y^2)
}

.log.pd.det <- function(x,dx=dim(x)[1]){
  y <- c(chol(x))[1L + 0L:(dx[1L] - 1L) * (dx[1L] + 1L)]
  sum(log(y^2))
}

.colmult <- function(v, M){
  t.default(v*t.default(M))
}

.vMMt <- function(v,M){
  M%*% (v*t.default(M))
}

.rowcol2idx <- function(y.dim,r.idx, c.idx=r.idx)
## > rowcol2idx(4,1:3,c(1,3:4))
##      [,1] [,2] [,3]
## [1,]    1    9   13
## [2,]    2   10   14
## [3,]    3   11   15
{
  ofs <-(1:y.dim-1)*y.dim
  outer(r.idx, ofs[c.idx], FUN="+")
}



## Get the generator type ("discrete", "continuous", "mixed")
##
.genType <- function(dd, cc){
  if (length(dd)==0){
    if (length(cc)>0){
      "continuous"
    } else {
      stop("Generator must have a discrete and/or continuous part\n")
    }
  } else {
    if (length(cc)>0){
      "mixed"
    } else {
      "discrete"
    }
  }
}


.infoPrint <- function(details, limit=1, ...,  prefix='.'){
  if(details>=limit){
    cat(paste(rep(prefix, limit), collapse=''), "")
    cat(...)
  }
}

.infoPrint2 <- function(details, limit=1, fmt, ...,  prefix='.'){
  if (details>=limit)
    cat(paste(paste(rep(prefix, limit), collapse=''),  sprintf(fmt, ...), collapse=' '))
}

.colstr <- function(x, collapse=" ")
  paste(x, collapse=collapse)



## .getModelType <- function(object){
##   if (is.null(object$parms$Sigma) && is.null(object$parms$K)){
##     "discrete"
##   } else {
##     if (is.null(object$parms$p) && is.null(object$parms$g)){
##       "continuous"
##     } else {
##       "mixed"
##     }
##   }
## }


## logLik.mModel <- function(object,...){
##   ans <- object$fitinfo$logL
##   attr(ans, "df") <- object$dimension["df"]
##   class(ans) <- "logLik"
##   return(ans)
## }
