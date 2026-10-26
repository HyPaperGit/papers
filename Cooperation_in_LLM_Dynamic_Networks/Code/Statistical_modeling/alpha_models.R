suppressPackageStartupMessages({
  library(glmmTMB)
})
args<-commandArgs(trailingOnly=TRUE)
stopifnot(length(args)>=3)
base<-normalizePath(args[1],winslash='/',mustWork=TRUE)
slug<-args[2]
outdir<-args[3]
panels<-if(length(args)>3)strsplit(args[4],',',fixed=TRUE)[[1]] else c('A','C','B')
stopifnot(all(panels %in% c('A','B','C')))
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)

gradient<-function(f)max(abs(f$obj$gr(f$fit$par)))
good<-function(f){
  V<-tryCatch(vcov(f)$cond,error=function(e)matrix(NA,1,1))
  isTRUE(f$sdr$pdHess)&&f$fit$convergence==0&&is.finite(gradient(f))&&gradient(f)<.002&&
    all(is.finite(V))&&all(diag(V)>0)&&all(is.finite(fixef(f)$cond))
}

fit_one<-function(panel,model){
  stem<-file.path(outdir,paste0(slug,'_',panel,'_M',model))
  if(file.exists(paste0(stem,'_diagnostics.csv'))&&read.csv(paste0(stem,'_diagnostics.csv'))$Status=='VERIFIED'){
    message('RESUME ',basename(stem));return(invisible(TRUE))
  }
  source<-file.path(base,switch(panel,A='overall.csv',B='static_ties.csv',C='dynamic_ties.csv'))
  d<-read.csv(source,colClasses=c(Network_ID='character',Participant_ID='character',AlphaCode='character'))
  ref<-if(panel=='C')'a10' else 'a00'
  lev<-if(panel=='B')sprintf('a%02d',0:9) else sprintf('a%02d',0:10)
  d$Alpha<-relevel(factor(d$AlphaCode,levels=lev),ref=ref)
  d$Network_ID<-factor(d$Network_ID);d$Participant_ID<-factor(d$Participant_ID)
  stopifnot(all(tapply(as.character(d$Network_ID),d$Participant_ID,function(x)length(unique(x)))==1))
  stopifnot(nlevels(d$Network_ID)==if(panel=='B')190 else 210)
  if(model==1)rhs<-~Alpha else {
    for(ai in 1:9)d[[paste0('I_a',sprintf('%02d',ai))]]<-d$I*as.integer(d$AlphaCode==sprintf('a%02d',ai))
    rhs<-reformulate(c('Alpha',if(panel!='B')'I',sprintf('I_a%02d',1:9)))
  }
  form<-as.formula(paste('cbind(Y,Trials-Y)',paste(deparse(rhs),collapse=' '),'+(1|Network_ID)+(1|Participant_ID)'))
  X<-model.matrix(rhs,d)
  stopifnot(qr(X)$rank==ncol(X),!any(c('Round','Degree','KDz','KSz') %in% all.vars(form)))
  cellvars<-if(model==1)'AlphaCode' else c('AlphaCode','I')
  cells<-aggregate(d[c('Y','Trials')],d[cellvars],sum)
  separation<-any(cells$Y==0|cells$Y==cells$Trials)
  cells$Eta<-qlogis((cells$Y+.5)/(cells$Trials+1))
  cellkey<-function(x)do.call(paste,c(x[cellvars],sep=':'))
  eta<-cells$Eta[match(cellkey(d),cellkey(cells))]
  startbeta<-as.numeric(lm.fit(X,eta)$coefficients)
  stopifnot(all(is.finite(startbeta)))
  kind<-if(separation)'MAP' else 'ML'
  priors<-if(kind=='MAP')data.frame(prior=c('normal(0,20)','normal(0,5)','gamma(10,2)'),
    class=c('fixef','fixef','ranef_sd'),coef=c('(Intercept)','','')) else NULL
  attempts<-list();fits<-list()
  run<-function(opt,polish=FALSE,start_from=NULL){
    tag<-paste(opt,if(polish)'polish' else 'initial',sep='_')
    st<-if(is.null(start_from))list(beta=startbeta,theta=log(c(.3,.6))) else
      list(beta=as.numeric(fixef(start_from)$cond),theta=unname(start_from$fit$par[names(start_from$fit$par)=='theta']))
    control<-if(opt=='nlminb')glmmTMBControl(optCtrl=list(iter.max=6000,eval.max=6000,rel.tol=1e-11),parallel=1L) else
      glmmTMBControl(optimizer=optim,optArgs=list(method='BFGS'),optCtrl=list(maxit=6000,reltol=if(polish)1e-14 else 1e-12),parallel=1L)
    warnings<-character();beg<-proc.time()[['elapsed']]
    message('START ',slug,' panel=',panel,' Model ',model,' ',kind,' ',tag,' rows=',nrow(d))
    f<-tryCatch(withCallingHandlers(glmmTMB(form,data=d,family=binomial('logit'),control=control,start=st,priors=priors),
      warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)e)
    if(inherits(f,'error'))stop(conditionMessage(f))
    ok<-good(f)
    attempts[[tag]]<<-data.frame(Attempt=tag,Method=kind,Convergence=f$fit$convergence,PDHess=f$sdr$pdHess,
      Gradient=gradient(f),Objective=f$fit$objective,Elapsed=proc.time()[['elapsed']]-beg,
      Acceptable=ok,Warnings=paste(warnings,collapse=' | '))
    fits[[tag]]<<-f
    write.csv(do.call(rbind,attempts),paste0(stem,'_attempts.csv'),row.names=FALSE)
    message('DONE attempt ',basename(stem),' ',tag,' acceptable=',ok,' objective=',signif(f$fit$objective,10),' gradient=',signif(gradient(f),3))
  }
  run('nlminb');run('BFGS')
  acceptable<-function()names(attempts)[vapply(attempts,function(z)isTRUE(z$Acceptable),logical(1))]
  finite<-names(fits)[vapply(fits,function(f)is.finite(f$fit$objective),logical(1))]
  if(!length(finite))stop('No finite objective: ',basename(stem))
  best<-finite[which.min(vapply(fits[finite],function(f)f$fit$objective,numeric(1)))]
  difference<-max(abs(fixef(fits[[finite[1]]])$cond-fixef(fits[[finite[length(finite)]]])$cond))
  if(!length(acceptable())||difference>.0005)run('BFGS',TRUE,fits[[best]])
  if(!length(acceptable())){
    saveRDS(list(fits=fits,formula=form),paste0(stem,'_unresolved.rds'))
    stop('Unresolved ',basename(stem),'; no estimator changed.')
  }
  chosen<-acceptable()[which.min(vapply(fits[acceptable()],function(f)f$fit$objective,numeric(1)))]
  f<-fits[[chosen]];b<-fixef(f)$cond;V<-vcov(f)$cond
  if(panel=='C'){
    targets<-c('(Intercept)',sprintf('Alphaa%02d',1:10),if(model==2)c('I',sprintf('I_a%02d',1:9)))
    L<-matrix(0,length(targets),length(b),dimnames=list(targets,names(b)))
    L['(Intercept)',c('(Intercept)','Alphaa00')]<-1
    for(ai in 1:10){
      term<-sprintf('Alphaa%02d',ai);L[term,'Alphaa00']<--1
      if(ai<10)L[term,term]<-1
    }
    if(model==2)for(term in c('I',sprintf('I_a%02d',1:9)))L[term,term]<-1
    b<-drop(L%*%b);V<-L%*%V%*%t(L)
  }
  fixed<-data.frame(LLM=slug,Panel=panel,Model=model,Term=names(b),Estimate=as.numeric(b),
    SE=sqrt(diag(V)),P=2*pnorm(abs(b/sqrt(diag(V))),lower.tail=FALSE),Method=kind)
  selected<-list()
  add<-function(label,alpha,terms){
    w<-setNames(rep(0,length(b)),names(b));w[terms]<-1
    e<-sum(w*b);se<-sqrt(drop(t(w)%*%V%*%w))
    selected[[paste(label,alpha)]]<<-data.frame(LLM=slug,Panel=panel,Model=model,Contrast=label,
      Alpha=alpha,Estimate=e,SE=se,P=2*pnorm(abs(e/se),lower.tail=FALSE),Method=kind,
      Actions=sum(d$Trials),Replications=nlevels(d$Network_ID))
  }
  for(ai in 1:9){
    a<-ai/10;term<-sprintf('Alphaa%02d',ai)
    add(if(model==1)'Pooled network comparison' else 'Without reputation comparison',a,term)
    if(model==2){
      rterms<-c(if(panel!='B')'I',sprintf('I_a%02d',ai))
      add('Reputation effect',a,rterms)
      add('With reputation comparison',a,c(term,rterms))
    }
  }
  if(panel!='B'){
    add(if(model==1)'Pooled network comparison' else 'Without reputation comparison',1,'Alphaa10')
    if(model==2){
      add('Reputation effect',1,'I')
      add('With reputation comparison',1,c('Alphaa10','I'))
    }
  }
  if(model==2&&panel!='B'){
    terms<-grep('^I_a',names(b),value=TRUE)
    W<-drop(t(b[terms])%*%solve(V[terms,terms,drop=FALSE],b[terms]))
    write.csv(data.frame(LLM=slug,Panel=panel,WaldChiSquare=W,DF=length(terms),P=pchisq(W,length(terms),lower.tail=FALSE),Method=kind),
      paste0(stem,'_interaction_test.csv'),row.names=FALSE)
  }
  sds<-vapply(VarCorr(f)$cond,function(v)attr(v,'stddev')[1],numeric(1))
  random<-data.frame(LLM=slug,Panel=panel,Model=model,Group=names(sds),Variance=sds^2,SD=sds)
  dg<-data.frame(LLM=slug,Panel=panel,Model=model,Status='VERIFIED',Method=kind,Chosen=chosen,
    Actions=sum(d$Trials),Cooperations=sum(d$Y),Replications=nlevels(d$Network_ID),Agents=nlevels(d$Participant_ID),
    GroupedRows=nrow(d),DesignColumns=ncol(X),DesignRank=qr(X)$rank,Gradient=gradient(f),
    PDHess=f$sdr$pdHess,Convergence=f$fit$convergence,SeparatedFixedCell=separation,Reference='a00',NativeReference=ref,
    Objective=f$fit$objective,InitialOptimizerCoefficientDifference=difference)
  write.csv(fixed,paste0(stem,'_fixed.csv'),row.names=FALSE)
  write.csv(V,paste0(stem,'_vcov.csv'),row.names=TRUE)
  write.csv(random,paste0(stem,'_random.csv'),row.names=FALSE)
  write.csv(do.call(rbind,selected),paste0(stem,'_contrasts.csv'),row.names=FALSE)
  write.csv(dg,paste0(stem,'_diagnostics.csv'),row.names=FALSE)
  saveRDS(list(fit=f,formula=form,rhs=rhs,model=model,panel=panel,method=kind,source=source,
    source_md5=unname(tools::md5sum(source)),reference=ref,reported_reference='a00',
    coefficients=b,vcov=V,chosen=chosen,priors=priors),paste0(stem,'.rds'))
  capture.output({print(form);print(fixed);print(random);print(dg)},file=paste0(stem,'_summary.txt'))
  message('VERIFIED ',basename(stem),' ',kind)
  rm(f,fits,d);invisible(gc());invisible(TRUE)
}
for(panel in panels)for(model in 1:2)fit_one(panel,model)
