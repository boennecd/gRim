#################################################################
###
### mmod   : General homogeneous mixed interaction models
###
#################################################################

#' @title Mixed interaction model.
#' 
#' @description A mixed interaction model is a model (often with conditional
#'     independence restrictions) for a combination of discrete and continuous
#'     variables.
#'
#' @name mmod
#' 
#' @aliases mmod coef.mModel coefficients.mModel print.mModel summary.mModel
#' mmod_dimension
#' 
#' @param formula A right hand sided formula specifying the model.
#' @param data Data (a dataframe)
#' @param marginal A possible subsets of columns of \code{data}; useful when
#'     \code{formula} contains model specification shortcuts.
#' @param fit Currently not used
#' @param details For printing debugging information
#' 
#' @return An object of class \code{mModel} and the more general class
#'     \code{iModel}.
#' @author Søren Højsgaard, \email{sorenh@@math.aau.dk}
#' @seealso \code{\link{dmod}} \code{\link{cmod}}
#' @keywords models
#' @examples
#' 
#' ### FIXME: To be written
#' 
#' @export mmod
mmod <- function(formula, data, marginal=NULL, fit=TRUE, details=0)
  {
    t.start <- proc.time()
    cl      <- match.call()
    data.names <- names(data)
    
    if (!is.null(marginal)) 
        marginal <- intersect(data.names, marginal)
    
    flist <- .pFormula2(formula, data.names, marginal) 
    ##v.sep = ":", g.sep = "+", ignore.power.value = FALSE)
    glist <- flist$glist
    
### Extract the relevant columns of the dataframe. Discrete variables
### appear to the left of continuous variables
    datainfo <- .MIdatainfo(flist$varNames, data)
    .infoPrint(details, "mmod: .disc.names :", datainfo$disc.names, "\n")
    .infoPrint(details, "mmod: .cont.names :", datainfo$cont.names, "\n")

    varNames  <- datainfo$data.names    
    res <- list(glist          = glist,
                varNames       = varNames,
                datainfo       = datainfo,
                fitinfo        = NULL,
                isFitted       = FALSE)
    
    upd <- .mModel_finalize(glist, varNames, datainfo)    
    res[names(upd)] <- upd
    class(res) <- c("mModel","iModel")

##    if (fit){
##      res <- fit(res)      # use fit.dmod directly
##    }
##    #cat("time elapsed since start:", (proc.time()-t.start)[1],"\n")            
##    res
##
    if (fit) fit(res) else res
  }


## FIXME .glistNUM is goodie; put somewhere
.glistNUM <- function(glist, varNames){
    lapply(glist, function(l) match(l, varNames))
}

.mModel_finalize <- function(glist, varNames, datainfo){

    zzz <- isGSD_glist(glist, discrete=datainfo$disc.names)
    glistNUM <- .glistNUM(glist, varNames)    
    modelinfo <- .mModelinfo(glist, datainfo)
    
    ret      <- list(glistNUM       = glistNUM,
                     modelinfo      = modelinfo,
                     properties     = zzz)
    ret
}


.mModelinfo <- function(glist, datainfo){
  .mModelinfoPrimitive(glist, datainfo$data.names, datainfo$disc.indic)
}

## Finds numeric representations of generators.
.mModelinfoPrimitive <- function(glist, data.names, disc.indic){

    #' cat(".mModelinfoPrimitive\n")
    #' str(list(glist=glist, data.names=data.names, disc.indic=disc.indic))
    
    len.glist     <- length( glist )
    n.disc.names  <- sum(disc.indic)

    disc.gen.num   <- lin.gen.num   <- quad.gen.num   <- list()
    disc.gen.names <- lin.gen.names <- quad.gen.names <- list()

    glist.disc <- glist.num.disc   <- vector("list", len.glist)
    glist.cont <- glist.num.cont   <- vector("list", len.glist)
    glist.num  <- vector("list", len.glist)

    disc.idx       <- lin.idx       <- quad.idx       <- 1
    i <- 1
    while ( i  <= len.glist){
        gen.names             <- glist[[ i ]]
        gen.num               <- match(gen.names,data.names)
        glist.num[[ i ]]      <- gen.num 
        glist.disc[[ i ]]     <- gen.names[disc.indic[gen.num]==1]
        glist.cont[[ i ]]     <- gen.names[disc.indic[gen.num]==0]
        glist.num.disc[[ i ]] <- gen.num[disc.indic[gen.num]==1]
        glist.num.cont[[ i ]] <- gen.num[disc.indic[gen.num]==0] - n.disc.names
        
        disc.num <- gen.num[gen.num <= n.disc.names]
        cont.num <- gen.num[gen.num >  n.disc.names]
        gentype  <- .genType(disc.num, cont.num)
        
        switch(gentype,
               "discrete"={
                   disc.gen.num[[disc.idx]]   <- disc.num
                   disc.gen.names[[disc.idx]] <- data.names[disc.num]
                   disc.idx <- disc.idx + 1
               },
               "continuous"={
                   quad.gen.num[[quad.idx]]   <- cont.num
                   quad.gen.names[[quad.idx]] <- data.names[cont.num]
                   quad.idx <- quad.idx + 1
               },
               "mixed"={
                   disc.gen.num[[disc.idx]]   <- disc.num
                   disc.gen.names[[disc.idx]] <- data.names[disc.num]
                   disc.idx <- disc.idx + 1
                   quad.gen.num[[quad.idx]]   <- cont.num
                   quad.gen.names[[quad.idx]] <- data.names[cont.num]
                   quad.idx <- quad.idx + 1
                   lin.num   <- vector("list", length(cont.num))
                   lin.names <- vector("list", length(cont.num))
                   for (k in seq_along(cont.num)){
                       zzz              <- c(cont.num[ k ], disc.num)
                       lin.num[[ k ]]   <- zzz
                       lin.names[[ k ]] <- data.names[zzz]
                   }
                   lin.gen.num[[lin.idx]]   <- lin.num
                   lin.gen.names[[lin.idx]] <- lin.names             
                   lin.idx <- lin.idx + 1
               })
        i = i + 1
    }

    lin.gen.num   <- unlist(lin.gen.num,   recursive=FALSE)
    lin.gen.names <- unlist(lin.gen.names, recursive=FALSE)
    
    dlq <- list(discrete  = removeRedundant( disc.gen.names ), 
                linear    = removeRedundant( lin.gen.names  ),  
                quadratic = removeRedundant( quad.gen.names ))  
    
    ans <- list(glist.num      = glist.num,
                glist.disc     = glist.disc,
                glist.cont     = glist.cont,
                glist.num.disc = glist.num.disc,
                glist.num.cont = glist.num.cont,              
                dlq=dlq)
    
    return(ans)
}


### Takes the vn-columns of data only and organises so that
### discrete appear before continuous
### Stops if a vn does not appear in data.
.MIdatainfo <- function(vn, dd)
  {
    ##cat(".MIdatainfo\n")
    data.names <- names(dd)
    
    zzz <- match(vn, data.names)
    if (any(is.na(zzz)))
      stop("variables: ", vn[is.na(zzz)], " are not in data\n")
        
    disc.indic <- 1*!c(lapply(dd, is.numeric),recursive=TRUE)
    used.indic <- rep.int(0, length(data.names))
    used.indic[zzz] <- 1
    
    used.disc <- used.indic * (disc.indic==1)
    used.cont <- used.indic * (disc.indic==0)
    
    if (sum(used.disc) == 0) stop("No discrete variables\n")
    if (sum(used.cont) == 0) stop("No continuous variables\n")
    
    select.col <- c(which(used.disc == 1), which(used.cont == 1))
    
    disc.indic2 <- rep.int(0, length(select.col))
    disc.indic2[1:sum(used.disc)] <- 1

    newdata <- dd[, select.col, drop=FALSE]
    lev <- lapply(newdata, levels)
    lev <- lev[lapply(lev, length) > 0]

    data.names <- names(newdata)
    disc.names <- data.names[disc.indic2 == 1]
    cont.names <- data.names[disc.indic2 == 0]

    ##CGstats <- .extendCGstats(CGstats(newdata, disc.names=disc.names, cont.names=cont.names,homogeneous=FALSE))
    CGstats <-
      .extendCGstats(CGstats_internal(newdata,
                                      disc.names=disc.names,
                                      cont.names=cont.names,
                                      homogeneous=FALSE))
    list(data=newdata,
         CGstats=CGstats,
         data.names=data.names,
         disc.names=disc.names,
         cont.names=cont.names,
         disc.indic=disc.indic2,
         n.disc.names = sum(disc.indic2),
         disc.levels=lev)    
  }

coef.mModel <- coefficients.mModel <- function(object,type="ghk", ...){
  type <- match.arg(type, c("ghk","pms"))
  val <- object$fitinfo$parms
  switch(type,
         "pms"={val<-ghk2pmsParms_(val)}
         )
  val
}


summary.mModel <- function(object, ...){
  .listprint <- function(z){
    for ( i  in 1:length(z)){
      cat(" {", i ,"} ")
      print(z[[ i ]])
    }
  }
  cat("Mixed interaction model: \n")
  cat("Generators:\n")
  utils::str(.glist(object), give.head=FALSE, no.list=TRUE, comp.str=" ")
  
  cat(sprintf("Discrete: %3i   Continuous: %3i\n",
              length(getmi(object, "disc.names")),
              length(getmi(object, "cont.names"))))
  cat(sprintf("Is graphical: %s  Is decomposable: %s\n",
              getmi(object, "isGraphical"),
              getmi(object, "isDecomposable")))
    
  cat(sprintf("Dimension: %3i df: %3i, independence df: %3i\n",
              object$dimension[1], object$dimension[4], object$dimension[5]))
  
  if (object$isFitted){
    cat(sprintf("logL: %f, iDeviance: %f \n",
                object$fitinfo$logL, 2*(object$fitinfo$logL-object$fitinfo$init.logL)))   
  }
    
  return(invisible(object))
}












