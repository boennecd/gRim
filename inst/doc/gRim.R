### R code from vignette source 'gRim.Rnw'
### Encoding: UTF-8

###################################################
### code chunk number 1: gRim.Rnw:25-28
###################################################
require( gRbase )
prettyVersion <- packageDescription("gRim")$Version
prettyDate <- format(Sys.Date())


###################################################
### code chunk number 2: gRim.Rnw:77-81
###################################################
dir.create("figures")
oopt <- options()
options("digits"=4, "width"=80)
options(useFancyQuotes="UTF-8")


###################################################
### code chunk number 3: gRim.Rnw:88-92
###################################################
options("width"=85)
library(gRim)
library(Rgraphviz)
ps.options(family="serif")


###################################################
### code chunk number 4: gRim.Rnw:124-127
###################################################
args(dmod)
args(cmod)
args(mmod)


###################################################
### code chunk number 5: gRim.Rnw:145-147
###################################################
data(reinis)
str(reinis)


###################################################
### code chunk number 6: gRim.Rnw:159-163
###################################################
data(reinis)
dm1<-dmod(list(c("smoke","systol"),c("smoke","mental","phys")), data=reinis)
dm1<-dmod(~smoke:systol + smoke:mental:phys, data=reinis)
dm1


###################################################
### code chunk number 7: gRim.Rnw:183-185
###################################################
formula(dm1)
str(terms(dm1))


###################################################
### code chunk number 8: gRim.Rnw:191-192
###################################################
summary(dm1)


###################################################
### code chunk number 9: gRim.Rnw:223-226
###################################################
dm2 <- dmod(~.^2, margin=c("smo","men","phy","sys"),
            data=reinis)
formula(dm2)


###################################################
### code chunk number 10: gRim.Rnw:230-233
###################################################
dm3 <- dmod(list(c("smoke", "systol"), c("smoke", "mental", "phys")),
            data=reinis, interactions=2)
formula(dm3)


###################################################
### code chunk number 11: gRim.Rnw:247-248
###################################################
iplot(dm1)


###################################################
### code chunk number 12: gRim.Rnw:262-266
###################################################
data(carcass)
cm1 <- cmod(~Fat11:Fat12:Fat13, data=carcass)
cm1 <- cmod(~Fat11:Fat12 + Fat12:Fat13 + Fat11:Fat13, data=carcass)
cm1


###################################################
### code chunk number 13: gRim.Rnw:272-273
###################################################
iplot(cm1)


###################################################
### code chunk number 14: gRim.Rnw:280-283
###################################################
data(milkcomp1)
mm1 <- mmod(~.^., data=milkcomp1)
mm1


###################################################
### code chunk number 15: gRim.Rnw:289-290
###################################################
iplot(mm1)


###################################################
### code chunk number 16: gRim.Rnw:307-309
###################################################
ms <- dmod(~.^., marginal=c("phys","mental","systol","family"), data=reinis)
formula(ms)


###################################################
### code chunk number 17: gRim.Rnw:315-317
###################################################
ms1 <- update(ms, list(dedge=~phys:mental))
formula(ms1)


###################################################
### code chunk number 18: gRim.Rnw:323-325
###################################################
ms2<- update(ms, list(dedge=~phys:mental+systol:family))
formula(ms2)


###################################################
### code chunk number 19: gRim.Rnw:331-333
###################################################
ms3 <- update(ms, list(dedge=~phys:mental:systol))
formula(ms3)


###################################################
### code chunk number 20: gRim.Rnw:339-341
###################################################
ms4 <- update(ms, list(dterm=~phys:mental:systol) )
formula(ms4)


###################################################
### code chunk number 21: gRim.Rnw:347-349
###################################################
ms5 <- update(ms, list(aterm=~phys:mental+phys:systol+mental:systol) )
formula(ms5)


###################################################
### code chunk number 22: gRim.Rnw:355-357
###################################################
ms6 <- update(ms, list(aedge=~phys:mental+systol:family))
formula(ms6)


###################################################
### code chunk number 23: gRim.Rnw:378-379
###################################################
cit <- ciTest(reinis, set=c("systol","smoke","family","phys"))


###################################################
### code chunk number 24: gRim.Rnw:412-413
###################################################
cit$slice


###################################################
### code chunk number 25: gRim.Rnw:445-446
###################################################
ciTest(reinis, set=c("systol","smoke","family","phys"), method='MC')


###################################################
### code chunk number 26: gRim.Rnw:457-459
###################################################
dm5 <- dmod(~ment:phys:systol+ment:systol:family+phys:systol:smoke,
            data=reinis)


###################################################
### code chunk number 27: fundamentalfig1
###################################################
iplot(dm5)


###################################################
### code chunk number 28: gRim.Rnw:484-486
###################################################
testdelete(dm5, ~smoke:systol)
testdelete(dm5, ~family:systol)


###################################################
### code chunk number 29: gRim.Rnw:505-506
###################################################
testadd(dm5, ~smoke:mental)


###################################################
### code chunk number 30: gRim.Rnw:532-533
###################################################
ed.in <- getInEdges(ugList(dm5$glist), type="decomposable")


###################################################
### code chunk number 31: gRim.Rnw:546-547
###################################################
ed.out <- getOutEdges(ugList(dm5$glist), type="decomposable")


###################################################
### code chunk number 32: gRim.Rnw:555-557
###################################################
args(testInEdges)
args(testOutEdges)


###################################################
### code chunk number 33: gRim.Rnw:571-573
###################################################
testInEdges(dm5, getInEdges(ugList(dm5$glist), type="decomposable"),
             k=log(sum(reinis)))


###################################################
### code chunk number 34: gRim.Rnw:595-598
###################################################
dm.sat <- dmod(~.^., data=reinis)
dm.back <- backward(dm.sat)
iplot(dm.back)


###################################################
### code chunk number 35: gRim.Rnw:613-616
###################################################
dm.i   <- dmod(~.^1, data=reinis)
dm.forw <- forward(dm.i)
iplot(dm.forw)


###################################################
### code chunk number 36: gRim.Rnw:680-684
###################################################
fix <- list(c("smoke","phys","systol"), c("systol","protein"))
fix <- do.call(rbind, unlist(lapply(fix, names2pairs),recursive=FALSE))
fix
dm.s3 <- backward(dm.sat, fixin=fix, details=1)


###################################################
### code chunk number 37: gRim.Rnw:695-696
###################################################
dm.i3 <- forward(dm.i, fixout=fix, details=1)


###################################################
### code chunk number 38: gRim.Rnw:902-903
###################################################
loglinGenDim(dm2$glist, reinis)


###################################################
### code chunk number 39: gRim.Rnw:957-960
###################################################
dm3 <- dmod(list(c("smoke", "systol"), c("smoke", "mental", "phys")),
            data=reinis)
names(dm3)


###################################################
### code chunk number 40: gRim.Rnw:968-969
###################################################
str(dm3$glist)


###################################################
### code chunk number 41: gRim.Rnw:973-974
###################################################
str(dm3$glistNUM)


###################################################
### code chunk number 42: gRim.Rnw:980-981
###################################################
dm3$varNames


###################################################
### code chunk number 43: gRim.Rnw:989-990
###################################################
str(dm3[c("varNames","conNames","conLevels")])


