#########################################################################
###
### Get edges *in* or *not in* a graph
###
### Input: Various formats
### Output: A p * 2 matrix
###
### getInEdges: if type="decomposable" then the edges returned are those
### edges e for which the graph is decomposable if e is removed. If
### type="unrestricted" then all edges are returned
###
### getOutEdges: if type="decomposable" then the edges returned are those
### edges e for which the graph is decomposable if e is added. If
### type="unrestricted" then all edges are returned
###
### Known issues: None
###
##########################################################################

#' @title Find edges in a graph and edges not in an undirected graph.
#' 
#' @description Returns the edges of a graph (or edges not in a graph) where the
#'     graph can be either a graphNEL object, a list of generators or an
#'     adjacency matrix.
#'
#' @name getEdges
#'
#' @details When \code{ingraph=TRUE}: If type="decomposable" then
#'     \code{getEdges()} returns those edges e for which the graph with e
#'     removed is decomposable.
#' 
#' When \code{ingraph=FALSE}: Likewise, if type="decomposable" then
#' \code{getEdges()} returns those edges e for which the graph with e added is
#' decomposable.
#' 
#' The functions \code{getInEdges()} and \code{getInEdges()} are just wrappers
#' for calls to \code{getEdges()}.
#' 
#' The workhorses are \code{getInEdgesMAT()} and \code{getOutEdgesMAT()} and
#' these work on adjacency matrices.
#' 
#' Regarding the argument \code{discrete}, please see the documentation of
#' \code{\link[gRbase]{mcsmarked}}.
#' 
#' @aliases getEdges getEdges.list getEdges.graphNEL getEdges.matrix getInEdges
#'     getOutEdges getEdgesMAT getInEdgesMAT getOutEdgesMAT
#' @param object An object representing a graph; either a generator list, a
#'     graphNEL object or an adjacency matrix.
#' @param type Either "unrestricted" or "decomposable"
#' @param ingraph If TRUE the result is the edges in the graph; if FALSE the
#'     result is the edges not in the graph.
#' @param discrete This argument is relevant only if \code{object} specifies a
#'     marked graph in which some vertices represent discrete variables and some
#'     represent continuous variables.
#' @param \dots Additional arguments; currently not used.
#' @return A p * 2 matrix with edges.
#' @note These functions work on undirected graphs. The behaviour is
#'     undocumented for directed graphs.
#' @author Søren Højsgaard, \email{sorenh@@math.aau.dk}
#' @seealso \code{\link{edgeList}}, \code{\link{nonEdgeList}},
#'     \code{\link{mcsmarked}}
#' @keywords utilities
#' @examples
#' 
#' gg     <- ug(~a:b:d + a:c:d + c:e)
#' glist  <- getCliques(gg)
#' adjmat <- as.adjMAT(gg)
#' 
#' #### On a glist
#' getEdges(glist)
#' getEdges(glist,type="decomposable")
#' # Deleting (a,d) would create a 4-cycle
#' 
#' getEdges(glist, ingraph=FALSE)
#' getEdges(glist,type="decomposable", ingraph=FALSE)
#' # Adding (e,b) would create a 4-cycle
#' 
#' #### On a graphNEL
#' getEdges(gg)
#' getEdges(gg,type="decomposable")
#' # Deleting (a,d) would create a 4-cycle
#' 
#' getEdges(gg, ingraph=FALSE)
#' getEdges(gg,type="decomposable", ingraph=FALSE)
#' # Adding (e,b) would create a 4-cycle
#' 
#' #### On an adjacency matrix
#' getEdges(adjmat)
#' getEdges(adjmat,type="decomposable")
#' # Deleting (a,d) would create a 4-cycle
#' 
#' getEdges(adjmat, ingraph=FALSE)
#' getEdges(adjmat,type="decomposable", ingraph=FALSE)
#' # Adding (e,b) would create a 4-cycle
#' 
#' 
#' ## Marked graphs; vertices a,b are discrete; c,d are continuous
#' UG <- ug(~a:b:c + b:c:d)
#' disc <- c("a","b")
#' getEdges(UG)
#' getEdges(UG, discrete=disc)
#' ## Above: same results; there are 5 edges in the graph
#' 
#' getEdges(UG, type="decomposable")
#' ## Above: 4 edges can be removed and will give a decomposable graph
#' ##(only removing the edge (b,c) would give a non-decomposable model)
#' 
#' getEdges(UG, type="decomposable", discrete=c("a","b"))
#' ## Above: 3 edges can be removed and will give a strongly decomposable
#' ## graph. Removing (b,c) would create a 4--cycle and removing (a,b)
#' ## would create a forbidden path; a path with only continuous vertices
#' ## between two discrete vertices.
#' 
#' 
#' @export getEdges
getEdges <- function(object, type="unrestricted", ingraph=TRUE, discrete=NULL, ...){
  UseMethod("getEdges")
}

getEdges.list <- function(object, type="unrestricted", ingraph=TRUE, discrete=NULL, ...){
    getEdgesMAT(glist2adjMAT(object), type=type, ingraph=ingraph, discrete=discrete, ...)
}

getEdges.graphNEL <- function(object, type="unrestricted", ingraph=TRUE, discrete=NULL, ...){
    getEdgesMAT(as.adjMAT(object), type=type, ingraph=ingraph, discrete=discrete, ...)
}

getEdges.matrix <- function(object, type="unrestricted", ingraph=TRUE, discrete=NULL, ...){
    getEdgesMAT(object, type=type, ingraph=ingraph, discrete=discrete, ...)    
}


getInEdges <- function(object, type="unrestricted", discrete=NULL, ...){
  getEdges(object, type=type, ingraph=TRUE, discrete=discrete, ...)
}

getOutEdges <- function(object, type="unrestricted", discrete=NULL, ...){
  getEdges(object, type=type, ingraph=FALSE, discrete=discrete, ...)
}


##########################################################################

getEdgesMAT <- function(adjmat, type="unrestricted", ingraph=TRUE, discrete=NULL, ...){
    if (ingraph)
        getInEdgesMAT(adjmat, type, discrete, ...)
    else
        getOutEdgesMAT(adjmat, type, discrete, ...)
}

getInEdgesMAT <- function(adjmat, type="unrestricted", discrete=NULL, ...){
  type <- match.arg(type, c("unrestricted", "decomposable"))
  emat <- edgeListMAT(adjmat, matrix=TRUE) #;  print(emat)
  if (type == "decomposable"){
      idx <- vector("logical", nrow(emat))    
      for (ii in seq_len(nrow(emat))){
        ed <- emat[ii, ] 
        adjmat[ed[1], ed[2]] <- adjmat[ed[2], ed[1]] <- 0L
        idx[ii] <- length(mcsmarkedMAT(adjmat, discrete=discrete)) > 0
        adjmat[ed[1], ed[2]] <- adjmat[ed[2], ed[1]] <- 1L
      }
      emat <- emat[idx, , drop=FALSE]
    }
  emat
}

getOutEdgesMAT <- function(adjmat, type="unrestricted", discrete=NULL, ...){
  type <- match.arg(type, c("unrestricted", "decomposable"))
  emat <- nonEdgeListMAT(adjmat, matrix=TRUE)
  if (type == "decomposable"){
      idx <- vector("logical", nrow(emat))    
      for (ii in seq_len(nrow(emat))){
        ed <- emat[ii,]
        adjmat[ed[1], ed[2]] <- adjmat[ed[2], ed[1]] <- 1L
        idx[ii] <- length(mcsmarkedMAT(adjmat, discrete=discrete)) > 0
        adjmat[ed[1], ed[2]] <- adjmat[ed[2], ed[1]] <- 0L
      }
      emat <- emat[idx, , drop=FALSE]
    }
  emat
}
