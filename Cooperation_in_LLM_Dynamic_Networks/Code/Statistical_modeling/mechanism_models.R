suppressPackageStartupMessages({
  library(glmmTMB)
})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)>=5)
panel_path <- args[1]
events_path <- args[2]
starts_path <- args[3]
base <- args[4]
slug <- args[5]
only <- if(length(args)>=6)args[6] else 'all'
selected_alpha <- if(length(args)>=7)as.numeric(args[7]) else NA_real_
stopifnot(only %in% c('all','reputation','model1','model2'))
dir.create(file.path(base,'fits'),recursive=TRUE,showWarnings=FALSE)
alphas <- if(is.na(selected_alpha))c(.5,.1,.2,.3,.4,.6,.7,.8,.9) else selected_alpha
initial <- read.csv(starts_path,stringsAsFactors=FALSE)
seed_values <- function(spec,kind){
  q<-initial[initial$Model==slug & initial$Specification==spec & initial$Kind==kind,]
  setNames(q$Value,q$Term)
}

weighted_sd <- function(x,w) {
  m<-weighted.mean(x,w);sqrt(sum(w*(x-m)^2)/(sum(w)-1))
}
grad <- function(f)max(abs(f$obj$gr(f$fit$par)))
is_good <- function(f) {
  V<-tryCatch(vcov(f)$cond,error=function(e)matrix(NA_real_,1,1))
  isTRUE(f$sdr$pdHess)&&f$fit$convergence==0&&is.finite(grad(f))&&grad(f)<.005&&
    all(is.finite(V))&&all(diag(V)>0)&&all(is.finite(fixef(f)$cond))
}

fit_one <- function(spec,alpha,all_data,source_path) {
  prefix<-file.path(base,'fits',paste0(slug,'_a',sprintf('%02d',round(alpha*10)),'_',spec))
  if(file.exists(paste0(prefix,'_diagnostics.csv'))) {
    prior<-read.csv(paste0(prefix,'_diagnostics.csv'))
    if(identical(prior$Status,'VERIFIED')&&file.exists(paste0(prefix,'.rds'))) {
      message('RESUME skip verified ',basename(prefix));return(invisible(TRUE))
    }
  }
  if(spec=='reputation') {
    d<-all_data[all_data$NetType %in% c('Static','Dynamic') |
      (all_data$NetType=='Mixed'&abs(all_data$Alpha-alpha)<1e-8),]
    d$D<-as.integer(d$NetType=='Dynamic')
    d$MD<-as.integer(d$NetType=='Mixed'&d$TieType=='Dynamic')
    d$MS<-as.integer(d$NetType=='Mixed'&d$TieType=='Static')
    d$I<-d$Reputation;d$MDI<-d$MD*d$I;d$MSI<-d$MS*d$I
    stopifnot(all(d$Trials>=1),all(!is.na(d$Action_Num)),
      all(d$Action_Num>=0&d$Action_Num<=d$Trials),
      all(d$Reputation[d$NetType=='Static']==0))
    weights<-d$Trials
    round_mean<-weighted.mean(d$Round,weights);round_sd<-weighted_sd(d$Round,weights)
    degree_mean<-weighted.mean(d$Number_of_Ties,weights);degree_sd<-weighted_sd(d$Number_of_Ties,weights)
    d$Rz<-(d$Round-round_mean)/round_sd
    d$Kz<-(d$Number_of_Ties-degree_mean)/degree_sd
    form<-cbind(Action_Num,Trials-Action_Num)~D+MD+MS+I+MDI+MSI+Rz+Kz+
      (1|Network_ID)+(1|Participant_ID)
    rhs<-~D+MD+MS+I+MDI+MSI+Rz+Kz
    seed_path<-starts_path
    expected_networks<-50
  } else {
    d<-all_data[all_data$NetType=='Dynamic'|
      (all_data$NetType=='Mixed'&abs(all_data$Alpha-alpha)<1e-8),]
    weights<-rep(1,nrow(d))
    degree_mean<-mean(d$Number_of_Ties);degree_sd<-sd(d$Number_of_Ties)
    d$Kz<-(d$Number_of_Ties-degree_mean)/degree_sd
    round_mean<-if(spec=='model1')mean(d$Round) else NA_real_
    round_sd<-if(spec=='model1')sd(d$Round) else NA_real_
    if(spec=='model1') {
      d$Rz<-(d$Round-round_mean)/round_sd
      form<-Action_Num~DynamicNetwork+Reputation+DynamicTie*PostUpdate+Rz+Kz+
        PriorNeighborhoodCooperation+(1|Network_ID)+(1|Participant_ID)
      rhs<-~DynamicNetwork+Reputation+DynamicTie*PostUpdate+Rz+Kz+PriorNeighborhoodCooperation
      stopifnot(all(d$Round %in% 2:15),all(d$PostUpdate==as.integer(d$Round %in% c(4,7,10,13))))
    } else {
      d$RF<-factor(d$Round,levels=c(4,7,10,13))
      form<-Action_Num~DynamicNetwork+Reputation+DynamicTie+RF+Kz+
        PriorNeighborhoodCooperation+LostTiePreviousUpdate+DynamicTie:LostTiePreviousUpdate+
        (1|Network_ID)+(1|Participant_ID)
      rhs<-~DynamicNetwork+Reputation+DynamicTie+RF+Kz+PriorNeighborhoodCooperation+
        LostTiePreviousUpdate+DynamicTie:LostTiePreviousUpdate
      stopifnot(all(d$Round %in% c(4,7,10,13)),all(d$PostUpdate==1))
    }
    stopifnot(all(d$Action_Num %in% c(0,1)),all(!is.na(d$PriorNeighborAction)),
      all(d$PriorNeighborhoodCooperation>=0&d$PriorNeighborhoodCooperation<=1))
    seed_path<-starts_path
    expected_networks<-40
  }
  for(v in c('Network_ID','Participant_ID'))d[[v]]<-factor(d[[v]])
  stopifnot(nlevels(d$Network_ID)==expected_networks,
    nrow(unique(d[c('Network_ID','NetType','Reputation','Alpha')]))==expected_networks)
  meta<-unique(d[c('Network_ID','NetType','Reputation')])
  cells<-table(meta$NetType,meta$Reputation)
  stopifnot(all(cells[c('Dynamic','Mixed'),c('0','1')]==10))
  if(spec=='reputation')stopifnot(cells['Static','0']==10,cells['Static','1']==0)
  X<-model.matrix(rhs,d)
  stopifnot(qr(X)$rank==ncol(X),!('Round_ID' %in% all.vars(form)),
            !('PriorNeighborAction' %in% all.vars(form)))
  cell_cols<-if(spec=='reputation')c('NetType','TieType','Reputation') else
    c('NetType','TieType','Reputation',if(spec=='model1')'PostUpdate' else 'LostTiePreviousUpdate')
  audit<-aggregate(cbind(Cooperation=d$Action_Num,Trials=weights),
                   d[cell_cols],sum)
  write.csv(audit,paste0(prefix,'_cells.csv'),row.names=FALSE)
  seed<-list(means=seed_values(spec,'mean'),scales=seed_values(spec,'scale'))
  beta0<-seed_values(spec,'beta')
  stopifnot(setequal(names(beta0),colnames(X)))
  beta0<-beta0[colnames(X)]
  old_kmean<-seed$means[['Number_of_Ties']]
  old_ksd<-seed$scales[['Number_of_Ties']]
  beta0['(Intercept)']<-beta0['(Intercept)']+beta0['Kz']*(degree_mean-old_kmean)/old_ksd
  beta0['Kz']<-beta0['Kz']*degree_sd/old_ksd
  if('Rz' %in% names(beta0)) {
    old_rmean<-seed$means[['Round']]
    old_rsd<-seed$scales[['Round']]
    beta0['(Intercept)']<-beta0['(Intercept)']+beta0['Rz']*(round_mean-old_rmean)/old_rsd
    beta0['Rz']<-beta0['Rz']*round_sd/old_rsd
  }
  old_sds<-seed_values(spec,'random_sd')
  stopifnot(identical(names(old_sds),c('Network_ID','Participant_ID')))
  start0<-list(beta=unname(beta0),theta=log(pmax(old_sds,.1)))
  attempts<-list();fits<-list()
  run<-function(optimizer,kind='ML',polish=FALSE) {
    tag<-paste(kind,optimizer,if(polish)'polish' else 'initial',sep='_')
    start<-start0
    if(polish&&length(fits)) {
      finite<-fits[vapply(fits,function(f)is.finite(f$fit$objective),logical(1))]
      best<-finite[[which.min(vapply(finite,function(f)f$fit$objective,numeric(1)))]]
      start<-list(beta=as.numeric(fixef(best)$cond),
                  theta=unname(best$fit$par[names(best$fit$par)=='theta']))
    }
    ctrl<-if(optimizer=='nlminb')glmmTMBControl(
      optCtrl=list(iter.max=15000,eval.max=15000,rel.tol=1e-11),parallel=1L) else
      glmmTMBControl(optimizer=optim,optArgs=list(method='BFGS'),
        optCtrl=list(maxit=15000,reltol=if(polish)1e-14 else 1e-12),parallel=1L)
    priors<-if(kind=='MAP')data.frame(
      prior=c('normal(0,20)','normal(0,5)','gamma(10,2)'),
      class=c('fixef','fixef','ranef_sd'),coef=c('(Intercept)','','')) else NULL
    warnings<-character();began<-proc.time()[['elapsed']]
    message('START ',slug,' alpha=',alpha,' ',spec,' ',tag,' actions=',sum(weights))
    fit<-tryCatch(withCallingHandlers(glmmTMB(form,data=d,family=binomial(link='logit'),
      control=ctrl,start=start,priors=priors),warning=function(w){
        warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)e)
    if(inherits(fit,'error')) {
      attempts[[tag]]<<-data.frame(Optimizer=optimizer,Method=kind,Polish=polish,
        Convergence=NA,PDHess=FALSE,Gradient=NA,Objective=NA,MaxBeta=NA,
        Elapsed=proc.time()[['elapsed']]-began,Acceptable=FALSE,Warnings=conditionMessage(fit))
      return(invisible(FALSE))
    }
    good<-is_good(fit)
    maxbeta<-max(abs(fixef(fit)$cond))
    if(spec=='reputation'&&kind=='ML'&&any(abs(fixef(fit)$cond[c('D','MD','MS','I','MDI','MSI')])>15))good<-FALSE
    attempts[[tag]]<<-data.frame(Optimizer=optimizer,Method=kind,Polish=polish,
      Convergence=fit$fit$convergence,PDHess=fit$sdr$pdHess,Gradient=grad(fit),
      Objective=fit$fit$objective,MaxBeta=maxbeta,Elapsed=proc.time()[['elapsed']]-began,
      Acceptable=good,Warnings=paste(warnings,collapse=' | '))
    fits[[tag]]<<-fit
    write.csv(do.call(rbind,attempts),paste0(prefix,'_attempts.csv'),row.names=FALSE)
    message('ATTEMPT ',slug,' alpha=',alpha,' ',spec,' ',tag,' good=',good,
      ' grad=',signif(grad(fit),4),' Hess=',fit$sdr$pdHess,' code=',fit$fit$convergence)
    invisible(good)
  }
  run('nlminb');run('BFGS')
  acceptable<-names(attempts)[vapply(attempts,function(a)isTRUE(a$Acceptable),logical(1))]
  if(!length(acceptable)) {
    run('BFGS',polish=TRUE)
    acceptable<-names(attempts)[vapply(attempts,function(a)isTRUE(a$Acceptable),logical(1))]
  }
  if(!length(acceptable)&&spec=='reputation') {
    run('nlminb','MAP');run('BFGS','MAP')
    acceptable<-names(attempts)[vapply(attempts,function(a)isTRUE(a$Acceptable),logical(1))]
  }
  write.csv(do.call(rbind,attempts),paste0(prefix,'_attempts.csv'),row.names=FALSE)
  if(!length(acceptable)) {
    saveRDS(list(fits=fits,formula=form,alpha=alpha,source_data=source_path),paste0(prefix,'_unresolved.rds'))
    write.csv(data.frame(LLM=slug,Alpha=alpha,Specification=spec,Status='NO_VERIFIED_ML_FIT',
      Rows=nrow(d),Actions=sum(weights),Replications=nlevels(d$Network_ID)),
      paste0(prefix,'_diagnostics.csv'),row.names=FALSE)
    message('UNRESOLVED ',basename(prefix));return(invisible(FALSE))
  }
  chosen<-acceptable[which.min(vapply(fits[acceptable],function(f)f$fit$objective,numeric(1)))]
  fit<-fits[[chosen]];method<-attempts[[chosen]]$Method
  beta<-fixef(fit)$cond;V<-vcov(fit)$cond;se<-sqrt(diag(V))
  fixed<-data.frame(LLM=slug,Alpha=alpha,Specification=spec,Term=names(beta),
    Estimate=as.numeric(beta),SE=as.numeric(se),Z=as.numeric(beta/se),
    P=2*pnorm(abs(beta/se),lower.tail=FALSE),Method=method,Actions=sum(weights))
  J<-diag(length(beta));dimnames(J)<-list(names(beta),names(beta))
  for(z in intersect(c('Rz','Kz'),names(beta))) {
    scale<-if(z=='Rz')round_sd else degree_sd
    ix<-fixed$Term==z;fixed$Estimate[ix]<-fixed$Estimate[ix]/scale;fixed$SE[ix]<-fixed$SE[ix]/scale
    fixed$Term[ix]<-if(z=='Rz')'Round' else 'Degree';J[z,z]<-1/scale
  }
  reportedV<-J%*%V%*%t(J);dimnames(reportedV)<-list(fixed$Term,fixed$Term)
  vc<-VarCorr(fit)$cond;sds<-vapply(vc,function(v)attr(v,'stddev')[1],numeric(1))
  random<-data.frame(Group=names(vc),Variance=sds^2,SD=sds)
  contrasts<-list()
  add<-function(label,terms) {
    w<-setNames(rep(0,length(beta)),names(beta));w[terms]<-1
    e<-sum(w*beta);s<-sqrt(drop(t(w)%*%V%*%w))
    contrasts[[label]]<<-data.frame(Contrast=label,Estimate=e,SE=s,
      P=2*pnorm(abs(e/s),lower.tail=FALSE),Lower=e-qnorm(.975)*s,Upper=e+qnorm(.975)*s)
  }
  if(spec=='reputation') {
    add('Mixed static vs Fully static, without reputation','MS')
    add('Reputation effect on Mixed static ties',c('I','MSI'))
    add('Mixed static with reputation vs Fully static without reputation',c('MS','I','MSI'))
    add('Static vs Fully dynamic reputation-effect difference','MSI')
    add('Reputation effect on Mixed dynamic ties',c('I','MDI'))
    add('Reputation effect in Fully dynamic networks','I')
  } else {
    event<-if(spec=='model1')'PostUpdate' else 'LostTiePreviousUpdate'
    add('Event association: static ties',event)
    add('Event association: dynamic ties',c(event,paste0('DynamicTie:',event)))
    add('Dynamic minus static event-association difference',paste0('DynamicTie:',event))
  }
  diagnostics<-data.frame(LLM=slug,Alpha=alpha,Specification=spec,Status='VERIFIED',Method=method,
    Chosen=chosen,Rows=nrow(d),Actions=sum(weights),Replications=nlevels(d$Network_ID),
    Agents=nlevels(d$Participant_ID),DesignColumns=ncol(X),DesignRank=qr(X)$rank,
    Gradient=grad(fit),PDHess=fit$sdr$pdHess,Convergence=fit$fit$convergence,
    Objective=fit$fit$objective,RoundMean=round_mean,RoundSD=round_sd,
    DegreeMean=degree_mean,DegreeSD=degree_sd,
    SourceMD5=unname(tools::md5sum(source_path)),SeedMD5=unname(tools::md5sum(seed_path)))
  comparisons<-do.call(rbind,lapply(names(fits),function(k){
    f<-fits[[k]];data.frame(Attempt=k,Method=attempts[[k]]$Method,
      ObjectiveDifference=f$fit$objective-fit$fit$objective,
      MaxBetaDifference=max(abs(fixef(f)$cond-beta)),Acceptable=attempts[[k]]$Acceptable)
  }))
  saveRDS(list(fit=fit,formula=form,alpha=alpha,specification=spec,method=method,
    source_data=source_path,source_md5=diagnostics$SourceMD5,seed_path=seed_path,
    degree_mean=degree_mean,degree_sd=degree_sd,round_mean=round_mean,round_sd=round_sd,
    chosen=chosen,reference=if(spec=='reputation')
      'MS: Fully static/no reputation; I: Fully dynamic; MSI: difference from Fully dynamic reputation effect'
      else if(spec=='model1')
      'Mixed network; no reputation; static ties; other analyzed rounds'
      else 'Mixed network; no reputation; static ties; no prior tie loss; round 4'),paste0(prefix,'.rds'))
  write.csv(fixed,paste0(prefix,'_fixed.csv'),row.names=FALSE)
  write.csv(random,paste0(prefix,'_random.csv'),row.names=FALSE)
  write.csv(V,paste0(prefix,'_vcov.csv'),row.names=TRUE)
  write.csv(reportedV,paste0(prefix,'_reported_vcov.csv'),row.names=TRUE)
  write.csv(do.call(rbind,contrasts),paste0(prefix,'_contrasts.csv'),row.names=FALSE)
  write.csv(diagnostics,paste0(prefix,'_diagnostics.csv'),row.names=FALSE)
  write.csv(comparisons,paste0(prefix,'_optimizer_comparison.csv'),row.names=FALSE)
  capture.output({print(form);print(fixed);print(random);print(diagnostics)},file=paste0(prefix,'_summary.txt'))
  message('DONE ',basename(prefix),' ',method,' verified; gradient=',signif(grad(fit),4))
  rm(seed,fit,fits,d);invisible(gc());invisible(TRUE)
}

for(spec in if(only=='all')c('reputation','model1','model2') else only) {
  source_path<-if(spec=='reputation')panel_path else events_path
  all_data<-read.csv(source_path,stringsAsFactors=FALSE,
    colClasses=c(Agent='character',Neighbor='character',Network_ID='character',Participant_ID='character'))
  if(spec=='reputation'){
    all_data<-all_data[all_data$Round %in% 13:15,]
    all_data$Trials<-1L
    stopifnot(all(!is.na(all_data$Action_Num)))
  } else if(spec=='model2')all_data<-all_data[all_data$Round %in% c(4,7,10,13),]

  for(alpha in alphas)tryCatch(fit_one(spec,alpha,all_data,source_path),error=function(e){
    message('ERROR ',slug,' ',spec,' alpha=',alpha,': ',conditionMessage(e))
    writeLines(conditionMessage(e),file.path(base,'fits',paste0(slug,'_a',sprintf('%02d',round(alpha*10)),'_',spec,'_error.txt')))
  })
  rm(all_data);invisible(gc())
}
