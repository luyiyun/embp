# install.packages("ggmagnify", repos = c("https://hughjonesd.r-universe.dev", 
#                                         "https://cloud.r-project.org"))
library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ncdf4)
library(tibble)
library(ggsci)
library(patchwork)
library(scales)
library(gg.gap)
library(cowplot)
library(ggmagnify)
library(ggfx)
library(grid)
#### functions ####

# twostage-conti
twostage_conti = function(data, nref, nstud){
  
  #### 1.  Create output matrix: returns betahax and its estimated variance
  ####     wts: betahatw, var(betahatw), inversevar, weights, 
  #             bhat, varbhat, betahatxi, varbetahatxi (not in that order)
  ####     ests:betahatx, var(betahatx), inversevar, weights
  output_ts           = matrix(NA, ncol=3, nrow=1)
  colnames(output_ts) = c("Point estimate of betax","Lower 95%CI", "Upper 95%CI")
  wts                 = matrix(NA, nrow=(nstud-nref), ncol=8) # for local labs
  ests                = matrix(NA, nrow=nref, ncol=4) # for reference labs
  
  
  #### 3. Get point and variance estimate of b_hat for each study in cal subset
  for(k in 1:(nstud-nref)){
    cal_data_s         = subset(data, S==k & H=="True")
    fit                = lm(X~W, data=cal_data_s)
    wts[k,1]    = fit$coefficients[2]
    wts[k,2]    = vcov(fit)[2,2]
  }
  
  #### 4. Compute betahatw, var(betahatw) for each study in noncal subset
  for(k in 1:(nstud-nref)){
    data_s          = subset(data, S==k) #study
    fitW            = lm(Y ~ W, data = data_s)
    wts[k,3] = fitW$coefficients[2] #betahatw
    wts[k,4] = vcov(fitW)[2,2] #var betahatw
  }
  
  #### 5. Regression calibration to get corrected pt/var estimates
  wts[,5] = wts[,3]/wts[,1] # betahat xi
  wts[,6] = wts[,4]/(wts[,1]^2) + (wts[,2]*wts[,3]^2)/(wts[,1]^4) #var betahat xi
  wts[,7] = 1/wts[,6] # var(betahat) xi ^ (-1)
  
  #### 6. Compute inverse variance weights
  total = sum(wts[,7])
  wts[,8] = wts[,7]/total
  
  #### 7. Compute final estimate of betahatx and var betahatxi from LOCAL studies
  betahatx_loc    = wts[,8]%*%wts[,5]
  varbetahatx_loc = (wts[,8]^2)%*%wts[,6]
  
  #### 8. Compute betahatx, var(betahatx) for studies using reference lab, if any
  if(nref>0){
    for(k in (nref+1):nstud){
      data_s        = subset(data, S==k) #study specific
      fitX          = lm(Y~X,data=data_s)
      ests[k-nref,1]     = fitX$coefficients[2] #betahatx
      ests[k-nref,2]     = vcov(fitX)[2,2] #var betahatx
      ests[k-nref,3]     = 1/ests[k-nref,2] #inverse var weight
    }
    
    
    #### 9. Inverse var weights to get one estimate from studies using reference lab
    total    = sum(ests[ ,3])
    ests[,4] = ests[,3]/total
    
    #### 10. Compute final estimate of betahatx and var betahatxi from reference lab studies
    betahatx_ref    = ests[,4]%*%ests[,1]
    varbetahatx_ref = (ests[,4]^2)%*%ests[,2]
    
    #### 11. Combine ref and loc estimates to provide final estimates of point, RR, var
    total = 1/varbetahatx_ref + 1/varbetahatx_loc
    w_ref = (1/varbetahatx_ref)/total
    w_loc = (1/varbetahatx_loc)/total
    
    betax_hat = betahatx_loc*w_loc + betahatx_ref*w_ref
    v = varbetahatx_ref*w_ref^2+varbetahatx_loc*w_loc^2
    
    output_ts[1] = betax_hat
    output_ts[2] = betax_hat-1.96*sqrt(v)
    output_ts[3] = betax_hat+1.96*sqrt(v)
    
    
    
  }else{ # estimates if only local studies were used
    output_ts[1] = betahatx_loc
    output_ts[2] = betahatx_loc-1.96*sqrt(varbetahatx_loc)
    output_ts[3] = betahatx_loc+1.96*sqrt(varbetahatx_loc)
  } 
  
  ##### 输出结果
  return(output_ts)
  
}

# twostage-binary
twostage_bi = function(data, nref, nstud){
  
  #### 1.  Create output matrix: returns betahax and its estimated variance
  ####     wts: betahatw, var(betahatw), inversevar, weights, 
  #             bhat, varbhat, betahatxi, varbetahatxi (not in that order)
  ####     ests:betahatx, var(betahatx), inversevar, weights
  output_ts           = matrix(NA, ncol=3, nrow=1)
  colnames(output_ts) = c("Point estimate of betax","Lower 95%CI", "Upper 95%CI")
  wts                 = matrix(NA, nrow=(nstud-nref), ncol=8) # for local labs
  ests                = matrix(NA, nrow=nref, ncol=4) # for reference labs
  
  
  #### 3. Get point and variance estimate of b_hat for each study in cal subset
  for(k in 1:(nstud-nref)){
    cal_data_s         = subset(data, S==k & H=="True")
    fit                = lm(X~W, data=cal_data_s)
    wts[k,1]    = fit$coefficients[2]
    wts[k,2]    = vcov(fit)[2,2]
  }
  
  #### 4. Compute betahatw, var(betahatw) for each study in noncal subset
  for(k in 1:(nstud-nref)){
    data_s          = subset(data, S==k) #study
    # fitW            = clogit(Y~W, data=data_s)
    fitW            = glm(Y~W, data=data_s,family = binomial(link = "logit"))
    wts[k,3] = fitW$coefficients[2] #betahatw
    wts[k,4] = vcov(fitW)[2,2] #var betahatw
  }
  
  #### 5. Regression calibration to get corrected pt/var estimates
  wts[,5] = wts[,3]/wts[,1] # betahat xi
  wts[,6] = wts[,4]/(wts[,1]^2) + (wts[,2]*wts[,3]^2)/(wts[,1]^4) #var betahat xi
  wts[,7] = 1/wts[,6] # var(betahat) xi ^ (-1)
  
  #### 6. Compute inverse variance weights
  total = sum(wts[,7])
  wts[,8] = wts[,7]/total
  
  #### 7. Compute final estimate of betahatx and var betahatxi from LOCAL studies
  betahatx_loc    = wts[,8]%*%wts[,5]
  varbetahatx_loc = (wts[,8]^2)%*%wts[,6]
  
  #### 8. Compute betahatx, var(betahatx) for studies using reference lab, if any
  if(nref>0){
    for(k in (nref+1):nstud){
      data_s        = subset(data, S==k) #study specific
      fitX          = glm(Y~X,data=data_s, family = binomial(link = "logit"))
      ests[k-nref,1]     = fitX$coefficients[2] #betahatx
      ests[k-nref,2]     = vcov(fitX)[2,2] #var betahatx
      ests[k-nref,3]     = 1/ests[k-nref,2] #inverse var weight
    }
    
    
    #### 9. Inverse var weights to get one estimate from studies using reference lab
    total    = sum(ests[ ,3])
    ests[,4] = ests[,3]/total
    
    #### 10. Compute final estimate of betahatx and var betahatxi from reference lab studies
    betahatx_ref    = ests[,4]%*%ests[,1]
    varbetahatx_ref = (ests[,4]^2)%*%ests[,2]
    
    #### 11. Combine ref and loc estimates to provide final estimates of point, RR, var
    total = 1/varbetahatx_ref + 1/varbetahatx_loc
    w_ref = (1/varbetahatx_ref)/total
    w_loc = (1/varbetahatx_loc)/total
    
    betax_hat = betahatx_loc*w_loc + betahatx_ref*w_ref
    v = varbetahatx_ref*w_ref^2+varbetahatx_loc*w_loc^2
    
    output_ts[1] = betax_hat
    output_ts[2] = betax_hat-1.96*sqrt(v)
    output_ts[3] = betax_hat+1.96*sqrt(v)
    
    
    
  }else{ # estimates if only local studies were used
    output_ts[1] = betahatx_loc
    output_ts[2] = betahatx_loc-1.96*sqrt(varbetahatx_loc)
    output_ts[3] = betahatx_loc+1.96*sqrt(varbetahatx_loc)
  } 
  
  ##### 输出结果
  return(output_ts)
  
}

# in-binary
int_bi <- function(data, nstud, nref){
  
  #### 1. Create matrix to store output #####
  output_int = matrix(NA, ncol=3, nrow=1)
  colnames(output_int) = c("Point estimate of betax","Lower 95%CI", "Upper 95%CI")
  #### 2. Compute other useful quantities #####
  np = dim(data)[1] # total number of subject
  n1 = c(rep(NA,nstud)) # number of subjects in each study
  nQ <- nstud-nref # number of studies needing calibration
  for(k in 1:nstud){
    n1[k] = sum(data$S==k)
  }
  #### 3. Complete calibration studies and add appropriate ahat, bhat to data frame #####
  a_hat = c(rep(NA,nQ), rep(0,nref))
  b_hat = c(rep(NA,nQ), rep(1,nref))
  ncal  = c(rep(NA,nstud))
  for(k in 1:nQ){
    cal_data_s  = subset(data, S==k & H=="True")
    ncal[k]     = nrow(cal_data_s)
    fit         = lm(X~W, data=cal_data_s)
    a_hat[k]    = fit$coefficients[1]
    b_hat[k]    = fit$coefficients[2]
  }
  data$a_hat = a_hat[data$S] ## adding a_hat and b_hat to the dataframe
  data$b_hat = b_hat[data$S]
  #### 4. Create xhat_fc and xhat_int variable- use H==2 to indicate when using X ref lab #####
  data$xhat_fc  = ifelse(data$H==2, data$X, data$a_hat + data$b_hat*data$W)
  data$xhat_int = ifelse(data$H=="False", data$xhat_fc, data$X) 
  #### 5 求beta值 #####
  #####5.1 设定beta0s和betax初始值 #######################################
  formula         = as.formula(paste("Y~xhat_int",sep="+"))
  int_fit         = glm(formula, data=data,family = binomial(link = "logit"))
  betax_init      = int_fit$coefficients[2]
  beta0_init      = rep(int_fit$coefficients[1],nstud)
  #####5.2 生成求解函数 #####
  funs <- function(beta0_hat, betax_hat){
    #### 5.2.1 phi_theta
    phi_beta0_hat = c(rep(0,nstud))
    phi_betax_sum <- 0
    for(l in 1:nstud){
      cal_data_s = subset(data, S==l) # 每个研究
      expBX     <- exp(beta0_hat[l]+betax_hat*cal_data_s$xhat_int)
      expitBX   <- expBX/(1+expBX)
      phi_beta0 <- cal_data_s$Y - expitBX
      phi_betax <- cal_data_s$xhat_int * phi_beta0
      
      phi_beta0_hat[l] <- sum(phi_beta0)
      phi_betax_sum <- phi_betax_sum+sum(phi_betax)
    }
    phi_theta <- c(phi_beta0_hat, phi_betax_sum)
    
    #### 5.2.2 hessian矩阵 
    H = matrix(0, ncol = nstud+1, nrow = nstud+1)
    for(l in 1:nstud){
      cal_data_s = subset(data, S==l) # 每个研究
      expBX     <- exp(beta0_hat[l]+betax_hat*cal_data_s$xhat_int)
      expitBX   <- expBX/(1+expBX)
      
      deri_phibeta0_beta0_s <- expitBX*(expitBX-1)
      deri_phibetax_beta0_s <- cal_data_s$xhat_int*deri_phibeta0_beta0_s
      deri_phibetax_betax_s <- cal_data_s$xhat_int*deri_phibetax_beta0_s
      
      H[l,l]           = sum(deri_phibeta0_beta0_s)         # beta0 beta0
      H[nstud+1,l]     = sum(deri_phibetax_beta0_s) # betax beta0
      
      H[l,nstud+1]     = H[nstud+1,l]
      
      H[nstud+1,nstud+1] = H[nstud+1,nstud+1]+
        sum(deri_phibetax_betax_s) # phi_betax_betax
    } 
    list(phi_theta = phi_theta, H = H)
  }
  #####5.3 牛顿法 #####
  Newtons <- function(funs, beta0_hat, betax_hat, ep = 1e-5, it_max = 100){
    index <- 0
    k <- 1
    beta <- c(beta0_hat, betax_hat)
    while (k <= it_max){
      beta1 <- beta
      obj <- funs(beta[1:nstud], beta[length(beta)])
      beta <- beta- solve(obj$H, obj$phi_theta)
      norm <- sqrt((beta - beta1) %*% (beta - beta1))
      if (norm < ep){
        index <- 1; break
      }
      k <- k + 1
    }
    obj <- funs(beta[1:nstud], beta[length(beta)])
    list(root = beta, it = k, index = index, FunVal = obj$phi_theta)
  }  
  logist_beta <- Newtons(funs, beta0_init, betax_init)
  # Obtain point estimate from logistic regression by Newtons
  beta0_hat <- logist_beta$root[1:nstud]
  betax_hat <- logist_beta$root[nstud+1]
  output_int[1] = betax_hat
  #### 6. Compute variance #######
  ##### 6.1 prepare matrices  #######################################
  dim_sand  = 2*nQ + nstud + 1
  A         = matrix(0, ncol = dim_sand, nrow = dim_sand)
  B         = matrix(0, ncol = dim_sand, nrow = dim_sand)
  ##### 6.2 A MATRIX #######################################
  ##### 6.2.1 Q及以内队列
  for(k in 1:nQ){
    cal_data_h0 = subset(data, (H=="False" & S==k)) # 用x~来估计x的子集，即需要校正的(部分校正方法)
    cal_data_h1 = subset(data, (H=="True" & S==k)) # 直接用x原值的子集(部分校正方法)
    expBX_h0     <- exp(beta0_hat[k]+betax_hat*cal_data_h0$xhat_int) # 看一下这些指标能不能直接计算出来
    expBX_h1     <- exp(beta0_hat[k]+betax_hat*cal_data_h1$xhat_int)
    expitBX_h0   <- expBX_h0/(1+expBX_h0)
    expitBX_h1   <- expBX_h1/(1+expBX_h1)
    phi_a     <- cal_data_h0$xhat_int-a_hat[k]-b_hat[k]*cal_data_h0$W
    phi_b     <- cal_data_h0$W*phi_a
    phi_beta0_h0 <- cal_data_h0$Y - expitBX_h0
    phi_beta0_h1 <- cal_data_h1$Y - expitBX_h1
    phi_betax_h0 <- cal_data_h0$xhat_int * phi_beta0_h0
    phi_betax_h1 <- cal_data_h1$xhat_int * phi_beta0_h1
    A[k, k]                    = sum(phi_a^2)                      # a a
    A[nQ+k, nQ+k]              = sum(phi_b^2)                      # b b
    A[2*nQ+k,2*nQ+k]           = sum(phi_beta0_h0^2) + 
      sum(phi_beta0_h1^2)               # beta0 beta0
    A[2*nQ+nstud+1,2*nQ+k]     = sum(phi_betax_h0*phi_beta0_h0)+
      sum(phi_betax_h1*phi_beta0_h1)    # betax beta0
    A[2*nQ+nstud+1,nQ+k]       = sum(phi_betax_h0*phi_b)     # betax b   
    A[2*nQ+nstud+1,k]          = sum(phi_betax_h0*phi_a)     # betax a
    A[2*nQ+k,nQ+k]             = sum(phi_beta0_h0*phi_b)     # beta0 b
    A[2*nQ+k,k]                = sum(phi_beta0_h0*phi_a)     # beta0 a
    A[nQ+k,k]                  = sum(phi_a*phi_b)           # b a 
    A[k, nQ+k]                = A[nQ+k, k]
    A[k,2*nQ+k]               = A[2*nQ+k,k]
    A[nQ+k,2*nQ+k]            = A[2*nQ+k,nQ+k]
    A[k,2*nQ+nstud+1]         = A[2*nQ+nstud+1,k]
    A[nQ+k,2*nQ+nstud+1]      = A[2*nQ+nstud+1,nQ+k]
    A[2*nQ+k,2*nQ+nstud+1]    = A[2*nQ+nstud+1,2*nQ+k]
    A[2*nQ+nstud+1,2*nQ+nstud+1] = A[2*nQ+nstud+1,2*nQ+nstud+1] + 
      sum(phi_betax_h0^2) + 
      sum(phi_betax_h1^2) # betax betax
  }  
  #### 6.2.2 不带Q队列
  if (nref > 0){
    for(l in (nQ+1):nstud){
      cal_data_s = subset(data, (H==2 & S==l)) # 不需要校正的研究
      expBX     <- exp(beta0_hat[l]+betax_hat*cal_data_s$xhat_int)
      expitBX   <- expBX/(1+expBX)
      phi_beta0 <- cal_data_s$Y - expitBX
      phi_betax <- cal_data_s$xhat_int * phi_beta0
      A[2*nQ+l,2*nQ+l]           = sum(phi_beta0^2)         # beta0 beta0
      A[2*nQ+nstud+1,2*nQ+l]     = sum(phi_betax*phi_beta0) # betax beta0
      A[2*nQ+l,2*nQ+nstud+1]     = A[2*nQ+nstud+1,2*nQ+l]
      A[2*nQ+nstud+1,2*nQ+nstud+1] = A[2*nQ+nstud+1,2*nQ+nstud+1] + 
        sum(phi_betax^2)  # betax betax
    } 
  }
  ##### 6.3 B Matrix ################################  
  #### 6.3.1 Q及以内队列
  for(k in 1:nQ){
    cal_data_h0 = subset(data, (H=="False" & S==k)) # 用x~来估计x的子集，即需要校正的(部分校正方法)
    cal_data_h1 = subset(data, (H=="True" & S==k)) # 直接用x原值的子集(部分校正方法)
    expBX_h0 <- exp(beta0_hat[k]+betax_hat*cal_data_h0$xhat_int)
    expBX_h1 <- exp(beta0_hat[k]+betax_hat*cal_data_h1$xhat_int)
    expitBX_h0   <- expBX_h0/(1+expBX_h0)
    expitBX_h1   <- expBX_h1/(1+expBX_h1)
    deri_phia_a <- rep(-1, nrow(cal_data_h0))
    deri_phib_a <- -cal_data_h0$W
    deri_phib_b <- -cal_data_h0$W^2
    deri_phibeta0_beta0_h0 <- expitBX_h0*(expitBX_h0-1)
    deri_phibeta0_beta0_h1 <- expitBX_h1*(expitBX_h1-1)
    deri_phibetax_beta0_h0 <- cal_data_h0$xhat_int*deri_phibeta0_beta0_h0
    deri_phibetax_beta0_h1 <- cal_data_h1$xhat_int*deri_phibeta0_beta0_h1
    deri_phibetax_betax_h0 <- cal_data_h0$xhat_int*deri_phibetax_beta0_h0
    deri_phibetax_betax_h1 <- cal_data_h1$xhat_int*deri_phibetax_beta0_h1
    deri_phibeta0_a <- betax_hat*deri_phibeta0_beta0_h0
    deri_phibeta0_b <- cal_data_h0$W*deri_phibeta0_a
    deri_phibetax_a <- (cal_data_h0$Y - expitBX_h0) + 
      cal_data_h0$xhat_int*deri_phibeta0_a
    deri_phibetax_b <- cal_data_h0$W*deri_phibetax_a
    B[k, k]                    = sum(deri_phia_a)             # phi_a_a
    B[nQ+k, nQ+k]              = sum(deri_phib_b)             # phi_b_b
    B[2*nQ+k,2*nQ+k]           = sum(deri_phibeta0_beta0_h0) + 
      sum(deri_phibeta0_beta0_h1)  # phi_beta0_beta0
    B[2*nQ+nstud+1,2*nQ+k]     = sum(deri_phibetax_beta0_h0)+
      sum(deri_phibetax_beta0_h1)  # phi_betax_beta0
    B[2*nQ+nstud+1,nQ+k]       = sum(deri_phibetax_b)         # phi_betax_b   
    B[2*nQ+nstud+1,k]          = sum(deri_phibetax_a)         # phi_betax_a
    B[2*nQ+k,nQ+k]             = sum(deri_phibeta0_b)         # phi_beta0_b
    B[2*nQ+k,k]                = sum(deri_phibeta0_a)         # phi_beta0_a
    B[nQ+k,k]                  = sum(deri_phib_a)             # phib_a 
    B[k, nQ+k]                = B[nQ+k, k]
    B[k,2*nQ+k]               = B[2*nQ+k,k]
    B[nQ+k,2*nQ+k]            = B[2*nQ+k,nQ+k]
    B[k,2*nQ+nstud+1]         = B[2*nQ+nstud+1,k]
    B[nQ+k,2*nQ+nstud+1]      = B[2*nQ+nstud+1,nQ+k]
    B[2*nQ+k,2*nQ+nstud+1]    = B[2*nQ+nstud+1,2*nQ+k]
    B[2*nQ+nstud+1,2*nQ+nstud+1] = B[2*nQ+nstud+1,2*nQ+nstud+1]+
      sum(deri_phibetax_betax_h0)+
      sum(deri_phibetax_betax_h1) # phi_betax_betax
  }  
  #### 6.3.2 不带Q队列
  if (nref > 0){
    for(l in (nQ+1):nstud){
      cal_data_s = subset(data, (H==2 & S==l)) # 同上，确定好用哪个数据集
      expBX     <- exp(beta0_hat[l]+betax_hat*cal_data_s$xhat_int)
      expitBX   <- expBX/(1+expBX)
      deri_phibeta0_beta0_s <- expitBX*(expitBX-1)
      deri_phibetax_beta0_s <- cal_data_s$xhat_int*deri_phibeta0_beta0_s
      deri_phibetax_betax_s <- cal_data_s$xhat_int*deri_phibetax_beta0_s
      B[2*nQ+l,2*nQ+l]           = sum(deri_phibeta0_beta0_s)         # beta0 beta0
      B[2*nQ+nstud+1,2*nQ+l]     = sum(deri_phibetax_beta0_s) # betax beta0
      B[2*nQ+l,2*nQ+nstud+1]     = B[2*nQ+nstud+1,2*nQ+l]
      B[2*nQ+nstud+1,2*nQ+nstud+1] = B[2*nQ+nstud+1,2*nQ+nstud+1]+
        sum(deri_phibetax_betax_s) # phi_betax_betax
    } 
  }
  ### 6.4 Compute sandwich variance estimator and place variance estimate in output
  V                = solve(B)%*%A%*%t(solve(B))
  # output_int[3]    = (1/np)*V[(2*nQ+nstud+1),(2*nQ+nstud+1)]
  output_int[2]    = betax_hat-1.96*sqrt(V[(2*nQ+nstud+1),(2*nQ+nstud+1)])
  output_int[3]    = betax_hat+1.96*sqrt(V[(2*nQ+nstud+1),(2*nQ+nstud+1)])
  #### 99. Return appropriate output
  return(output_int)
}

# fc-binary
fc_bi = function(data, nstud, nref){
  #### 1. Create matrix to store output #####
  output_fc = matrix(NA, ncol=3, nrow=1)
  colnames(output_fc) = c("Point estimate of betax","Lower 95%CI", "Upper 95%CI")
  #### 2. Compute other useful quantities #####
  np = dim(data)[1] # total number of subject
  n1 = c(rep(NA,nstud)) # number of subjects in each study
  nQ <- nstud-nref # number of studies needing calibration
  for(k in 1:nstud){
    n1[k] = sum(data$S==k)
  }
  #### 3. Complete calibration studies and add appropriate ahat, bhat to data frame #####
  a_hat = c(rep(NA,nQ), rep(0,nref))
  b_hat = c(rep(NA,nQ), rep(1,nref))
  ncal  = c(rep(NA,nstud))
  for(k in 1:nQ){
    cal_data_s  = subset(data, S==k & H=="True")
    ncal[k]     = nrow(cal_data_s)
    fit         = lm(X~W, data=cal_data_s)
    a_hat[k]    = fit$coefficients[1]
    b_hat[k]    = fit$coefficients[2]
  }
  data$a_hat = a_hat[data$S] ## adding a_hat and b_hat to the dataframe
  data$b_hat = b_hat[data$S]
  #### 4. Create xhat_fc and xhat_int variable- use H==2 to indicate when using X ref lab #####
  data$xhat_fc  = ifelse(data$H==2, data$X, data$a_hat + data$b_hat*data$W)
  # data$xhat_int = ifelse(data$H=="FALSE", data$xhat_fc, data$X) 
  #### 5 求beta值 #####
  #####5.1 设定beta0s和betax初始值 #######################################
  formula         = as.formula(paste("Y~xhat_fc",sep="+"))
  fc_fit         = glm(formula, data=data,family = binomial(link = "logit"))
  betax_init      = fc_fit$coefficients[2]
  beta0_init      = rep(fc_fit$coefficients[1],nstud)
  #####5.2 生成求解函数 #####
  funs <- function(beta0_hat, betax_hat){
    #### 5.2.1 phi_theta
    phi_beta0_hat = c(rep(0,nstud))
    phi_betax_sum <- 0
    for(l in 1:nstud){
      cal_data_s = subset(data, S==l) # 每个研究
      expBX     <- exp(beta0_hat[l]+betax_hat*cal_data_s$xhat_fc)
      expitBX   <- expBX/(1+expBX)
      phi_beta0 <- cal_data_s$Y - expitBX
      phi_betax <- cal_data_s$xhat_fc * phi_beta0
      
      phi_beta0_hat[l] <- sum(phi_beta0)
      phi_betax_sum <- phi_betax_sum+sum(phi_betax)
    }
    phi_theta <- c(phi_beta0_hat, phi_betax_sum)
    
    #### 5.2.2 hessian矩阵 
    H = matrix(0, ncol = nstud+1, nrow = nstud+1)
    for(l in 1:nstud){
      cal_data_s = subset(data, S==l) # 每个研究
      expBX     <- exp(beta0_hat[l]+betax_hat*cal_data_s$xhat_fc)
      expitBX   <- expBX/(1+expBX)
      
      deri_phibeta0_beta0_s <- expitBX*(expitBX-1)
      deri_phibetax_beta0_s <- cal_data_s$xhat_fc*deri_phibeta0_beta0_s
      deri_phibetax_betax_s <- cal_data_s$xhat_fc*deri_phibetax_beta0_s
      
      H[l,l]           = sum(deri_phibeta0_beta0_s)         # beta0 beta0
      H[nstud+1,l]     = sum(deri_phibetax_beta0_s) # betax beta0
      
      H[l,nstud+1]     = H[nstud+1,l]
      
      H[nstud+1,nstud+1] = H[nstud+1,nstud+1]+
        sum(deri_phibetax_betax_s) # phi_betax_betax
    } 
    list(phi_theta = phi_theta, H = H)
  }
  
  #####5.3 牛顿法 #####
  Newtons <- function(funs, beta0_hat, betax_hat, ep = 1e-5, it_max = 100){
    index <- 0
    k <- 1
    beta <- c(beta0_hat, betax_hat)
    while (k <= it_max){
      beta1 <- beta
      obj <- funs(beta[1:nstud], beta[length(beta)])
      beta <- beta- solve(obj$H, obj$phi_theta)
      norm <- sqrt((beta - beta1) %*% (beta - beta1))
      if (norm < ep){
        index <- 1; break
      }
      k <- k + 1
    }
    obj <- funs(beta[1:nstud], beta[length(beta)])
    list(root = beta, it = k, index = index, FunVal = obj$phi_theta)
  }  
  
  logist_beta <- Newtons(funs,beta0_init, betax_init)
  
  # Obtain point estimate from standard logistic regression by Newtons
  beta0_hat <- logist_beta$root[1:nstud]
  betax_hat <- logist_beta$root[nstud+1]
  output_fc[1]   = betax_hat
  
  #### 6. Compute variance####
  ##### 6.1 prepare matrices  #######################################
  dim_sand  = 2*nQ + nstud + 1
  A         = matrix(0, ncol = dim_sand, nrow = dim_sand)
  B         = matrix(0, ncol = dim_sand, nrow = dim_sand)
  
  ##### 6.2 A MATRIX #######################################
  #### 6.2.1 Q及以内队列
  for(k in 1:nQ){
    cal_data_s = subset(data, (H!=2 & S==k)) # 用x~来估计x的子集，即需要校正的(部分校正方法)
    expBX     <- exp(beta0_hat[k]+betax_hat*cal_data_s$xhat_fc) # 看一下这些指标能不能直接计算出来
    expitBX   <- expBX/(1+expBX)
    phi_a     <- cal_data_s$xhat_fc-a_hat[k]-b_hat[k]*cal_data_s$W
    phi_b     <- cal_data_s$W*phi_a
    phi_beta0 <- cal_data_s$Y - expitBX
    phi_betax <- cal_data_s$xhat_fc * phi_beta0
    
    A[k, k]                    = sum(phi_a^2)             # a a
    A[nQ+k, nQ+k]              = sum(phi_b^2)             # b b
    A[2*nQ+k,2*nQ+k]           = sum(phi_beta0^2)         # beta0 beta0
    
    A[2*nQ+nstud+1,2*nQ+k]     = sum(phi_betax*phi_beta0) # betax beta0
    A[2*nQ+nstud+1,nQ+k]       = sum(phi_betax*phi_b)     # betax b   
    A[2*nQ+nstud+1,k]          = sum(phi_betax*phi_a)     # betax a
    A[2*nQ+k,nQ+k]             = sum(phi_beta0*phi_b)     # beta0 b
    A[2*nQ+k,k]                = sum(phi_beta0*phi_a)     # beta0 a
    A[nQ+k,k]                  = sum(phi_a*phi_b)           # b a 
    
    A[k, nQ+k]                = A[nQ+k, k]
    A[k,2*nQ+k]               = A[2*nQ+k,k]
    A[nQ+k,2*nQ+k]            = A[2*nQ+k,nQ+k]
    A[k,2*nQ+nstud+1]         = A[2*nQ+nstud+1,k]
    A[nQ+k,2*nQ+nstud+1]      = A[2*nQ+nstud+1,nQ+k]
    A[2*nQ+k,2*nQ+nstud+1]    = A[2*nQ+nstud+1,2*nQ+k]
    A[2*nQ+nstud+1,2*nQ+nstud+1] = A[2*nQ+nstud+1,2*nQ+nstud+1] + 
      sum(phi_betax^2)
  }  
  
  #### 6.2.2 不带Q队列
  if (nref > 0){
    for(l in (nQ+1):nstud){
      cal_data_s = subset(data, (H==2 & S==l)) # 同上，确定好用哪个数据集
      expBX     <- exp(beta0_hat[l]+betax_hat*cal_data_s$xhat_fc)
      expitBX   <- expBX/(1+expBX)
      phi_beta0 <- cal_data_s$Y - expitBX
      phi_betax <- cal_data_s$xhat_fc * phi_beta0
      
      A[2*nQ+l,2*nQ+l]           = sum(phi_beta0^2)         # beta0 beta0
      
      A[2*nQ+nstud+1,2*nQ+l]     = sum(phi_betax*phi_beta0) # betax beta0
      
      A[2*nQ+l,2*nQ+nstud+1]     = A[2*nQ+nstud+1,2*nQ+l]
      
      A[2*nQ+nstud+1,2*nQ+nstud+1] = A[2*nQ+nstud+1,2*nQ+nstud+1] + 
        sum(phi_betax^2)    
    } 
  }
  
  ##### 6.3 B Matrix ################################  
  #### 6.3.1 Q及以内队列
  for(k in 1:nQ){
    cal_data_s = subset(data, (H!=2 & S==k)) # 
    
    expBX_s <- exp(beta0_hat[k]+betax_hat*cal_data_s$xhat_fc)
    expitBX_s   <- expBX_s/(1+expBX_s)
    
    deri_phia_a <- rep(-1, nrow(cal_data_s))
    deri_phib_a <- -cal_data_s$W
    deri_phib_b <- -cal_data_s$W^2
    
    deri_phibeta0_beta0_s <- expitBX_s*(expitBX_s-1)
    deri_phibetax_beta0_s <- cal_data_s$xhat_fc*deri_phibeta0_beta0_s
    deri_phibetax_betax_s <- cal_data_s$xhat_fc*deri_phibetax_beta0_s
    
    deri_phibeta0_a <- betax_hat*deri_phibeta0_beta0_s
    deri_phibeta0_b <- cal_data_s$W*deri_phibeta0_a
    deri_phibetax_a <- (cal_data_s$Y - expitBX_s) + 
      cal_data_s$xhat_fc*deri_phibeta0_a
    deri_phibetax_b <- cal_data_s$W*deri_phibetax_a
    
    B[k, k]                    = sum(deri_phia_a)             # phi_a_a
    B[nQ+k, nQ+k]              = sum(deri_phib_b)             # phi_b_b
    
    B[2*nQ+k,2*nQ+k]           = sum(deri_phibeta0_beta0_s)   # phi_beta0_beta0
    
    B[2*nQ+nstud+1,2*nQ+k]     = sum(deri_phibetax_beta0_s)  # phi_betax_beta0
    
    B[2*nQ+nstud+1,nQ+k]       = sum(deri_phibetax_b)         # phi_betax_b   
    B[2*nQ+nstud+1,k]          = sum(deri_phibetax_a)         # phi_betax_a
    B[2*nQ+k,nQ+k]             = sum(deri_phibeta0_b)         # phi_beta0_b
    B[2*nQ+k,k]                = sum(deri_phibeta0_a)         # phi_beta0_a
    B[nQ+k,k]                  = sum(deri_phib_a)             # phib_a 
    
    
    B[k, nQ+k]                = B[nQ+k, k]
    B[k,2*nQ+k]               = B[2*nQ+k,k]
    B[nQ+k,2*nQ+k]            = B[2*nQ+k,nQ+k]
    B[k,2*nQ+nstud+1]         = B[2*nQ+nstud+1,k]
    B[nQ+k,2*nQ+nstud+1]      = B[2*nQ+nstud+1,nQ+k]
    B[2*nQ+k,2*nQ+nstud+1]    = B[2*nQ+nstud+1,2*nQ+k]
    
    B[2*nQ+nstud+1,2*nQ+nstud+1] = B[2*nQ+nstud+1,2*nQ+nstud+1]+
      sum(deri_phibetax_betax_s) # phi_betax_betax
  }  
  
  #### 6.3.2 不带Q队列
  if (nref>0){
    for(l in (nQ+1):nstud){
      cal_data_s = subset(data, (H==2 & S==l)) # 同上，确定好用哪个数据集
      expBX     <- exp(beta0_hat[l]+betax_hat*cal_data_s$xhat_fc)
      expitBX   <- expBX/(1+expBX)
      
      deri_phibeta0_beta0_s <- expitBX*(expitBX-1)
      deri_phibetax_beta0_s <- cal_data_s$xhat_fc*deri_phibeta0_beta0_s
      deri_phibetax_betax_s <- cal_data_s$xhat_fc*deri_phibetax_beta0_s
      
      B[2*nQ+l,2*nQ+l]           = sum(deri_phibeta0_beta0_s)         # beta0 beta0
      B[2*nQ+nstud+1,2*nQ+l]     = sum(deri_phibetax_beta0_s) # betax beta0
      
      B[2*nQ+l,2*nQ+nstud+1]     = B[2*nQ+nstud+1,2*nQ+l]
      
      B[2*nQ+nstud+1,2*nQ+nstud+1] = B[2*nQ+nstud+1,2*nQ+nstud+1]+
        sum(deri_phibetax_betax_s) # phi_betax_betax
    } 
  }
  
  
  ### 6.4 Compute sandwich variance estimator and place variance estimate in output
  V                = solve(B)%*%A%*%t(solve(B))
  
  output_fc[2]    = betax_hat-1.96*sqrt(V[(2*nQ+nstud+1),(2*nQ+nstud+1)])
  output_fc[3]    = betax_hat+1.96*sqrt(V[(2*nQ+nstud+1),(2*nQ+nstud+1)])
  
  #### 99. Return appropriate output
  return(output_fc)
}

###################data process##################################
# summary-table
data_clean_conti = function(data, sheet){
  df <- read_excel(data, sheet = sheet)
  key_cols <- df[-1, 1:2]
  colnames(key_cols) <- c("Sample_size", "Beta_X")
  mse_df <- df[-1, 7:10] %>%
    mutate(Metric = "MSE") %>%
    rename(EMBP = 1, Naive = 2, X_only = 3, Two_stage = 4)%>%
    mutate(EMBP = as.numeric(EMBP)/100,
           Naive = as.numeric(Naive)/100,
           X_only = as.numeric(X_only)/100,
           Two_stage = as.numeric(Two_stage)/100)
  cov_df <- df[-1, 11:14] %>%
    mutate(Metric = "Coverage(%)") %>%
    rename(EMBP = 1, Naive = 2, X_only = 3, Two_stage = 4)%>%
    mutate(EMBP = as.numeric(EMBP),
           Naive = as.numeric(Naive),
           X_only = as.numeric(X_only),
           Two_stage = as.numeric(Two_stage))
  df_long <- bind_rows(mse_df, cov_df) %>%
    mutate(row_id = row_number()) 
  final_df <- bind_cols(key_cols[rep(1:nrow(key_cols), 2), ], df_long) %>%
    pivot_longer(cols = c("EMBP", "Naive", "X_only", "Two_stage"),
                 names_to = "Method", values_to = "Value")%>%
    mutate(Sample_size = case_when(
      Sample_size == 100 ~ "Sample size=100",
      Sample_size == 200 ~ "Sample size=200",
      TRUE ~ as.character(Sample_size)
    ))
}
data_clean_bi = function(data, sheet){
  df <- read_excel(data, sheet = sheet)
  key_cols <- df[-1, 1:2]
  colnames(key_cols) <- c("Sample_size", "OR")
  mse_df <- df[-1, 9:14] %>%
    mutate(Metric = "MSE") %>%
    rename(EMBP = 1, Naive = 2, X_only = 3, Two_stage = 4, Internalized = 5, Full_calibration = 6)%>%
    mutate(EMBP = as.numeric(EMBP)/100,
           Naive = as.numeric(Naive)/100,
           X_only = as.numeric(X_only)/100,
           Two_stage = as.numeric(Two_stage)/100,
           Internalized = as.numeric(Internalized)/100,
           Full_calibration = as.numeric(Full_calibration)/100)
  cov_df <- df[-1, 15:20] %>%
    mutate(Metric = "Coverage(%)") %>%
    rename(EMBP = 1, Naive = 2, X_only = 3, Two_stage = 4, Internalized = 5, Full_calibration = 6)%>%
    mutate(EMBP = as.numeric(EMBP),
           Naive = as.numeric(Naive),
           X_only = as.numeric(X_only),
           Two_stage = as.numeric(Two_stage),
           Internalized = as.numeric(Internalized),
           Full_calibration = as.numeric(Full_calibration))
  df_long <- bind_rows(mse_df, cov_df) %>%
    mutate(row_id = row_number()) 
  final_df <- bind_cols(key_cols[rep(1:nrow(key_cols), 2), ], df_long) %>%
    pivot_longer(cols = c("EMBP", "Naive", "X_only", "Two_stage", "Internalized", "Full_calibration"),
                 names_to = "Method", values_to = "Value") %>%
    mutate(Sample_size = case_when(
      Sample_size == 100 ~ "Sample size=100",
      Sample_size == 200 ~ "Sample size=200",
      TRUE ~ as.character(Sample_size)
    ))
}

# bias-j
data_bias_conti=function(dir, betax,samplesize,sigma){
  final_bias <- NULL
  for (n in 1:length(betax)){
    cat("n=",n,"--sample size:",samplesize[n],"--betax:",betax[n], "\n")
    ana_file <- paste0(dir,"ana_continue_wo_z_",samplesize[n],"_", sigma, "_",betax[n],"/analyzed_results.nc")
    data_file <- paste0(dir,"data_continue_wo_z_",samplesize[n],"_", sigma, "_",betax[n],"/data.csv")
    # 读取embp,x-only,naive
    nc <- nc_open(ana_file)
    embp_est <- ncvar_get(nc, "embp")[1,13,]
    embp_bias <-  embp_est-as.numeric(betax[n])
    xonly_est <- ncvar_get(nc, "xonly")[1,13,]
    xonly_bias <-  xonly_est-as.numeric(betax[n])
    naive_est <- ncvar_get(nc, "naive")[1,13,]
    naive_bias <-  naive_est-as.numeric(betax[n])
    # twostage分析
    data <- read.csv(data_file)
    results <- NULL
    for (m in 1:1000){
      data2 <- subset(data, repeat.==m-1)
      # data2 <- data %>%
      #   filter(repeat. == (m - 1), S %in% c(1, 2, 3))
      result <- twostage_conti(data=data2, nref=0, nstud=4)
      # result <- twostage_conti(data=data2, nref=0, nstud=3)
      results <- rbind(results,result[1])
    }
    twostage_bias <- as.numeric(results-as.numeric(betax[n]))
    # 合并数据集
    df_bias <- bind_rows(
      tibble(Value = embp_bias,  Method = "EMBP", Sample_size = paste0("Sample size=",samplesize[n]), Beta_X = betax[n]),
      tibble(Value = xonly_bias, Method = "X_only", Sample_size = paste0("Sample size=",samplesize[n]), Beta_X = betax[n]),
      tibble(Value = naive_bias, Method = "Naive", Sample_size = paste0("Sample size=",samplesize[n]), Beta_X = betax[n]),
      tibble(Value = twostage_bias, Method = "Two_stage", Sample_size = paste0("Sample size=",samplesize[n]), Beta_X = betax[n])
      # tibble(Value = naive_bias, Method = "Naive", Sample_size = paste0("Sample size=",samplesize[n]), Beta_X = betax[n])
    )
    final_bias <- rbind(final_bias,df_bias)
  }
  return(final_bias)
}
data_bias_bi=function(dir, OR,samplesize,prevalence){
  final_bias <- NULL
  for (n in 1:length(betax)){
    cat("n=",n,"--sample size:",samplesize[n],"--OR:",OR[n], "\n")
    ana_file <- paste0(dir,"lap/binary_wo_z_",samplesize[n],"_", prevalence, "_",as.numeric(OR[n]),"/analyzed_results.nc")
    data_file <- paste0(dir,"data/binary_wo_z_",samplesize[n],"_", prevalence, "_",as.numeric(OR[n]),"/data.csv")
    # 读取embp,x-only,naive
    nc <- nc_open(ana_file)
    embp_est <- ncvar_get(nc, "embp")[1,13,]
    embp_bias <-  embp_est-log(as.numeric(OR[n]))
    xonly_est <- ncvar_get(nc, "xonly")[1,13,]
    xonly_bias <-  xonly_est-log(as.numeric(OR[n]))
    naive_est <- ncvar_get(nc, "naive")[1,13,]
    naive_bias <-  naive_est-log(as.numeric(OR[n]))
    data <- read.csv(data_file)
    ints <- tss <- fcs <- NULL
    for (m in 1:1000){
      data2 <- subset(data, repeat.==m-1)
      # data2 <- data %>%
      #   filter(repeat. == (m - 1), S %in% c(1, 2, 3))
      # twostage分析
      ts <- twostage_bi(data=data2, nref=0, nstud=4)
      int <- int_bi(data=data2, nref=0, nstud=4)
      fc <- fc_bi(data=data2, nref=0, nstud=4)
      # ts <- twostage_bi(data=data2, nref=0, nstud=3)
      # int <- int_bi(data=data2, nref=0, nstud=3)
      # fc <- fc_bi(data=data2, nref=0, nstud=3)
      tss <- rbind(tss,ts[1])
      ints <- rbind(ints,int[1])
      fcs <- rbind(fcs,fc[1])
    }
    twostage_bias <- as.numeric(tss-log(as.numeric(OR[n])))
    internal_bias <- as.numeric(ints-log(as.numeric(OR[n])))
    fullcali_bias <- as.numeric(fcs-log(as.numeric(OR[n])))
    # 合并数据集
    df_bias <- bind_rows(
      tibble(Value = embp_bias,  Method = "EMBP", Sample_size = paste0("Sample size=",samplesize[n]), OR = OR[n]),
      tibble(Value = xonly_bias, Method = "X_only", Sample_size = paste0("Sample size=",samplesize[n]), OR = OR[n]),
      tibble(Value = naive_bias, Method = "Naive", Sample_size = paste0("Sample size=",samplesize[n]), OR = OR[n]),
      tibble(Value = twostage_bias, Method = "Two_stage", Sample_size = paste0("Sample size=",samplesize[n]), OR = OR[n]),
      tibble(Value = twostage_bias, Method = "Internalized", Sample_size = paste0("Sample size=",samplesize[n]), OR = OR[n]),
      tibble(Value = twostage_bias, Method = "Full_calibration", Sample_size = paste0("Sample size=",samplesize[n]), OR = OR[n])
      # tibble(Value = naive_bias, Method = "Naive", Sample_size = paste0("Sample size=",samplesize[n]), OR = OR[n])
    )
    final_bias <- rbind(final_bias,df_bias)
  }
  return(final_bias)
}

####################plot#################################
# box plot-data_bias
box_plot_conti = function(data, title){
  data$Method <- factor(data$Method, levels = c("EMBP",  "Naive", "X_only", "Two_stage"))
  ggplot(data, aes(x = Beta_X, y = Value, color = Method)) +
    geom_boxplot(outlier.size = 0.8, width = 0.6) +
    geom_jitter(aes(color = Method),
                position = position_jitterdodge(jitter.width = 0.4, dodge.width = 0.6),
                size = 0.3, alpha = 0.1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.3) +
    facet_wrap(~ Sample_size, ncol = 1, scales = "free_y", strip.position = "right") +
    scale_fill_npg() +
    scale_color_npg() +
    labs(x = NULL, y = NULL, title = title) +
    theme_bw() +
    theme(# 图例
      legend.position = "left",
      legend.text = element_text(size = 8), 
      legend.title = element_text(size = 10),
      # 整图大标题
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      # 整图x标题
      axis.title.x = element_text(size = 8),
      axis.title.y = element_text(size = 8),
      # 子图刻度
      axis.ticks.length = unit(2, "pt"),
      axis.ticks.x = element_line(linewidth = 0.3, color = "black"),
      axis.ticks.y = element_line(linewidth = 0.3, color = "black"),
      axis.text.x = element_text(size = 6),
      axis.text.y = element_text(size = 6),
      # 横纵框文字
      strip.text = element_text(size = 7, margin = margin(2, 2, 2, 2)),
    )
}
box_plot_bi = function(data, title){
  data$Method <- factor(data$Method, levels = c("EMBP",  "Naive", "X_only", "Two_stage", "Internalized", "Full_calibration"))
  ggplot(data, aes(x = OR, y = Value, color = Method)) +
    # geom_boxplot(outlier.size = 0.8, width = 0.6) +
    # geom_jitter(aes(color = Method),
    #             position = position_jitterdodge(jitter.width = 0.6, dodge.width = 0.6),
    #             size = 0.3, alpha = 0.1) +
    geom_boxplot(outlier.size = 0.4, width = 0.51) +
    geom_jitter(aes(color = Method),
                position = position_jitterdodge(jitter.width = 0.3, dodge.width = 0.51),
                size = 0.2, alpha = 0.1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.3) +
    facet_wrap(~ Sample_size, ncol = 1, scales = "free_y", strip.position = "right") +
    # scale_x_continuous(
    #   limits = c(1, 3.25),
    #   breaks = seq(1.25, 3, by = 0.25)
    # ) +
    scale_fill_npg() +
    scale_color_npg() +
    labs(x = NULL, y = NULL, title = title) +
    theme_bw() +
    theme(# 图例
      legend.position = "left",
      legend.text = element_text(size = 8), 
      legend.title = element_text(size = 10),
      # 整图大标题
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      # 整图x标题
      axis.title.x = element_text(size = 8),
      axis.title.y = element_text(size = 8),
      # 子图刻度
      axis.ticks.length = unit(2, "pt"),
      axis.ticks.x = element_line(linewidth = 0.3, color = "black"),
      axis.ticks.y = element_line(linewidth = 0.3, color = "black"),
      axis.text.x = element_text(size = 6),
      axis.text.y = element_text(size = 6),
      # 横纵框文字
      strip.text = element_text(size = 7, margin = margin(2, 2, 2, 2)),
    )
}

# line plot-summary_data
line_plot_conti = function(data){
  data$Method <- factor(data$Method, levels = c("EMBP",  "Naive", "X_only", "Two_stage"))
  ggplot(data, aes(x = as.numeric(Beta_X), y = Value, color = Method, linetype = Method)) +
    geom_line(size = 0.5) +
    geom_point(size = 1) +
    scale_x_continuous(
      limits = c(0.3, 4.2),
      breaks = seq(0.5, 4.0, by = 0.5)
    ) +
    facet_wrap(~ Sample_size, ncol = 1, scales = "free_y", strip.position = "right") +
    scale_color_npg() +
    scale_linetype_manual(values = c(
      "EMBP" = "solid", 
      "Naive" = "dashed", 
      "X_only" = "dotdash", 
      "Two_stage" = "twodash"
    )) +
    labs(x = NULL, y = NULL, title = NULL) +
    theme_bw() +
    theme(legend.position = "left",
          legend.text = element_text(size = 8), 
          legend.title = element_text(size = 10),
          axis.ticks.length = unit(2, "pt"),
          axis.ticks.x = element_line(linewidth = 0.3, color = "black"),
          axis.text.x = element_text(size = 6),
          axis.ticks.y = element_line(linewidth = 0.3, color = "black"),
          axis.text.y = element_text(size = 6),
          strip.text = element_text(size = 7, margin = margin(2, 2, 2, 2))
        )
}
line_plot_bi = function(data){
  data$Method <- factor(data$Method, levels = c("EMBP",  "Naive", "X_only", "Two_stage", "Internalized", "Full_calibration"))
  ggplot(data, aes(x = as.numeric(OR), y = Value, color = Method, linetype = Method)) +
    geom_line(size = 0.5) +
    geom_point(size = 1) +
    scale_x_continuous(
      limits = c(1.2, 3.05),
      breaks = seq(1.25, 3, by = 0.25)
    ) +
    facet_wrap(~ Sample_size, ncol = 1, scales = "free_y", strip.position = "right") +
    scale_color_npg() +
    scale_linetype_manual(values = c(
      "EMBP" = "solid", 
      "Naive" = "dashed", 
      "X_only" = "dotdash", 
      "Two_stage" = "twodash",
      "Internalized" = "1212", 
      "Full_calibration" = "longdash"
    )) +
    labs(x = NULL, y = NULL, title = NULL) +
    theme_bw() +
    theme(legend.position = "left",
          legend.text = element_text(size = 8), 
          legend.title = element_text(size = 10),
          axis.ticks.length = unit(2, "pt"),
          axis.ticks.x = element_line(linewidth = 0.3, color = "black"),
          axis.text.x = element_text(size = 6),
          axis.ticks.y = element_line(linewidth = 0.3, color = "black"),
          axis.text.y = element_text(size = 6),
          strip.text = element_text(size = 7, margin = margin(2, 2, 2, 2))
    )
}

# conti-2002、2003、2012、2013、4001##############################################
# binary-3001、3002、3004、3005、4002##############################################
data <- "test.xlsx"
s2002_sum <- data_clean_conti(data, 1)
s3001_sum <- data_clean_bi(data, 5)
s3002_sum <- data_clean_bi(data, 6)
s3004_sum <- data_clean_bi(data, 7)
s3005_sum <- data_clean_bi(data, 8)
s4001_sum <- data_clean_conti(data, 9)
s4002_sum <- data_clean_bi(data, 10)

s2002_mse <- s2002_sum %>% filter(Metric == "MSE")
s2002_cov <- s2002_sum %>% filter(Metric == "Coverage(%)")
s2003_mse <- s2003_sum %>% filter(Metric == "MSE")
s2003_cov <- s2003_sum %>% filter(Metric == "Coverage(%)")
s2012_mse <- s2012_sum %>% filter(Metric == "MSE")
s2012_cov <- s2012_sum %>% filter(Metric == "Coverage(%)")
s2013_mse <- s2013_sum %>% filter(Metric == "MSE")
s2013_cov <- s2013_sum %>% filter(Metric == "Coverage(%)")
s3001_mse <- s3001_sum %>% filter(Metric == "MSE")
s3001_cov <- s3001_sum %>% filter(Metric == "Coverage(%)")
s3002_mse <- s3002_sum %>% filter(Metric == "MSE")
s3002_cov <- s3002_sum %>% filter(Metric == "Coverage(%)")
s3004_mse <- s3004_sum %>% filter(Metric == "MSE")
s3004_cov <- s3004_sum %>% filter(Metric == "Coverage(%)")
s3005_mse <- s3005_sum %>% filter(Metric == "MSE")
s3005_cov <- s3005_sum %>% filter(Metric == "Coverage(%)")
s4001_mse <- s4001_sum %>% filter(Metric == "MSE")
s4001_cov <- s4001_sum %>% filter(Metric == "Coverage(%)")
s4002_mse <- s4002_sum %>% filter(Metric == "MSE")
s4002_cov <- s4002_sum %>% filter(Metric == "Coverage(%)")


# bias
setwd("E:/SongJL/016_research/EMBP/bayesian_biomark_pooling/experiments/embp/")
betax <- rep(c("0.5","1.0","1.5","2.0","2.5","3.0","3.5","4.0"),2)
OR <- rep(c("1.25", "1.50", "1.75", "2.00", "2.25", "2.50", "2.75", "3.00"),2)
samplesize  <- c(rep(100,8),rep(200,8))
s2002_bias <- data_bias_conti("scenario2002/", betax,samplesize,10)
s2003_bias <- data_bias_conti("scenario2003/", betax,samplesize,10)
s2012_bias <- data_bias_conti("scenario2012/", betax,samplesize,10)
s2013_bias <- data_bias_conti("scenario2013/", betax,samplesize,5)
s3001_bias <-  data_bias_bi("scenario3001/", OR,samplesize,0.3)
s3002_bias <-  data_bias_bi("scenario3002/", OR,samplesize,0.5)
s3004_bias <-  data_bias_bi("scenario3004/", OR,samplesize,0.3)
s3005_bias <-  data_bias_bi("scenario3005/", OR,samplesize,0.3)
s4001_bias <- data_bias_conti("scenario4001/", betax,samplesize,10)
s4002_bias <-  data_bias_bi("scenario4002/", OR,samplesize,0.3)

# bias-绘制箱式图
box_s2002 <- box_plot_conti(s2002_bias, NULL)
box_s2003 <- box_plot_conti(s2003_bias, NULL)
box_s2012 <- box_plot_conti(s2012_bias, NULL)
box_s2013 <- box_plot_conti(s2013_bias, NULL)
box_s3001 <- box_plot_bi(s3001_bias, NULL)
box_s3002 <- box_plot_bi(s3002_bias, NULL)
box_s3004 <- box_plot_bi(s3004_bias, NULL)
box_main_s3004 <- box_plot_bi(s3004_bias, NULL)
box_s3005 <- box_plot_bi(s3005_bias, NULL)

box_s4001 <- box_plot_conti(s4001_bias, NULL)
box_s4002 <- box_plot_bi(s4002_bias, NULL)

s3004_bias_zoom <- s3004_bias %>%
  group_by(Method, Sample_size, OR) %>%  
  filter(Value < 2 & Sample_size == "Sample size=100") %>%
  ungroup()
s3004_bias_zoom$Method <- factor(s3004_bias_zoom$Method,
                                 levels = c("EMBP",  "Naive", "X_only", "Two_stage", "Internalized", "Full_calibration"))
box_zoom_s3004 <- ggplot(s3004_bias_zoom, aes(x = OR, y = Value, color = Method)) +
  geom_boxplot(outlier.size = 0.4, width = 0.51) +
  geom_jitter(aes(color = Method),
              position = position_jitterdodge(jitter.width = 0.3, dodge.width = 0.51),
              size = 0.2, alpha = 0.1)  +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.3) +
  scale_fill_npg() +
  scale_color_npg() +
  labs(x = NULL, y = NULL, title = NULL) +
  theme_bw() +
  theme(legend.position = "none",
    axis.ticks.length = unit(2, "pt"),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_line(linewidth = 0.3, color = "black"),
    axis.text.x = element_blank(),
    axis.text.y = element_text(size = 10),
    plot.margin = unit(c(0.1, 0.1, 0.1, 0.1), "cm") 
  ) + 
  annotation_custom(
    grob = rectGrob(gp = gpar(fill = NA, col = "gray40", lwd = 0.2)),
    xmin = -Inf, xmax = Inf,
    ymin = -Inf, ymax = Inf
  )

# MSE
line_mse_s2002 <- line_plot_conti(s2002_mse)
line_mse_s2003 <- line_plot_conti(s2003_mse)
line_mse_s2012 <- line_plot_conti(s2012_mse)
line_mse_s2013 <- line_plot_conti(s2013_mse)
line_mse_s3001 <- line_plot_bi(s3001_mse)
line_mse_s3002 <- line_plot_bi(s3002_mse)
line_main_s3004 <- line_plot_bi(s3004_mse)# sample size = 100
line_mse_s3005 <- line_plot_bi(s3005_mse)

line_mse_s4001 <- line_plot_conti(s4001_mse)
line_mse_s4002 <- line_plot_bi(s4002_mse)

# zoom
inset = line_main_s3004 + geom_line(size = 0.1) +
  geom_point(size = 0.2) +
  theme(axis.ticks.length = unit(1, "pt"),
        axis.text.y = element_text(size = 4),
        axis.ticks.y = element_line(linewidth = 0.2, color = "black"))+
  inset_theme(axes = "y", margin=1)
line_mse_s3004 <- line_main_s3004 + geom_magnify(plot = inset,
                                                 expand = 0,
                                                 recompute = TRUE,
                                                 colour = "white",
                                                 shadow = FALSE,
                                                 linewidth = 0.2,
                                                 scale.inset = c(1,1),
                                                 from = c(1.2, 3.05, -0.1, 1), 
                                                 to = c(1.2, 2.1 ,750, 1750))
inset = line_main_s4004 + geom_line(size = 0.1) +
  geom_point(size = 0.2) +
  theme(axis.ticks.length = unit(1, "pt"),
        axis.text.y = element_text(size = 4),
        axis.ticks.y = element_line(linewidth = 0.2, color = "black"))+
  inset_theme(axes = "y", margin=1)
line_mse_s4004 <- line_main_s4004 + geom_magnify(plot = inset,
                                                 expand = 0,
                                                 recompute = TRUE,
                                                 colour = "gray40",
                                                 shadow = FALSE,
                                                 linewidth = 0.2,
                                                 scale.inset = c(1,1),
                                                 from = c(1.2, 3.05, -0.1, 1), 
                                                 to = c(1.2, 2.1 ,10, 30))
# coverage_rate
line_main_s2002 <- line_plot_conti(s2002_cov)
line_main_s2003 <- line_plot_conti(s2003_cov)
line_main_s2012 <- line_plot_conti(s2012_cov)
line_main_s2013 <- line_plot_conti(s2013_cov)
line_cov_s3001 <- line_plot_bi(s3001_cov)
line_cov_s3002 <- line_plot_bi(s3002_cov)
line_cov_s3004 <- line_plot_bi(s3004_cov) 
line_cov_s3005 <- line_plot_bi(s3005_cov)

line_main_s4001 <- line_plot_conti(s4001_cov)
line_cov_s4002 <- line_plot_bi(s4002_cov)

# zoom
inset = line_main_s2002 + 
        theme(axis.ticks.length = unit(1, "pt"),
              axis.text.y = element_text(size = 4),
              axis.ticks.y = element_line(linewidth = 0.2, color = "black"))+
        inset_theme(axes = "y", margin=1)
line_cov_s2002 <- line_main_s2002 + geom_magnify(plot = inset,
                                                 expand = 0.1,
                                                 recompute = TRUE,
                                                 colour = "gray40",
                                                 shadow = FALSE,
                                                 linewidth = 0.2,
                                                 scale.inset = c(1,0.3),
                                                 from = c(0.5, 4, 89.5, 98.5), 
                                                 to = c(0.38, 4.0 ,25, 70))

inset = line_main_s2003 + 
  theme(axis.ticks.length = unit(1, "pt"),
        axis.text.y = element_text(size = 4),
        axis.ticks.y = element_line(linewidth = 0.2, color = "black"))+
  inset_theme(axes = "y", margin=1)
line_cov_s2003 <- line_main_s2003 + geom_magnify(plot = inset,
                                                 expand = 0.1,
                                                 recompute = TRUE,
                                                 colour = "gray40",
                                                 shadow = FALSE,
                                                 linewidth = 0.2,
                                                 scale.inset = c(1,0.3),
                                                 from = c(0.5, 4, 89.5, 98.5), 
                                                 to = c(0.38, 4.0 ,25, 70))

inset = line_main_s2012 + 
  theme(axis.ticks.length = unit(1, "pt"),
        axis.text.y = element_text(size = 4),
        axis.ticks.y = element_line(linewidth = 0.2, color = "black"))+
  inset_theme(axes = "y", margin=1)
line_cov_s2012 <- line_main_s2012 + geom_magnify(plot = inset,
                                                 expand = 0.1,
                                                 recompute = TRUE,
                                                 colour = "gray40",
                                                 shadow = FALSE,
                                                 linewidth = 0.2,
                                                 scale.inset = c(1,0.3),
                                                 from = c(0.5, 4, 91, 99), 
                                                 to = c(0.38, 4.0 ,25, 70))

inset = line_main_s2013 + 
  theme(axis.ticks.length = unit(1, "pt"),
        axis.text.y = element_text(size = 4),
        axis.ticks.y = element_line(linewidth = 0.2, color = "black"))+
  inset_theme(axes = "y", margin=1)
line_cov_s2013 <- line_main_s2013 + geom_magnify(plot = inset,
                                                 expand = 0.1,
                                                 recompute = TRUE,
                                                 colour = "gray40",
                                                 shadow = FALSE,
                                                 linewidth = 0.2,
                                                 scale.inset = c(1,0.3),
                                                 from = c(0.5, 4, 89.25, 98.25), 
                                                 to = c(0.38, 4.0 ,25, 70))

inset = line_main_s4001 + 
  theme(axis.ticks.length = unit(1, "pt"),
        axis.text.y = element_text(size = 4),
        axis.ticks.y = element_line(linewidth = 0.2, color = "black"))+
  inset_theme(axes = "y", margin=1)
line_cov_s4001 <- line_main_s4001 + geom_magnify(plot = inset,
                                                 expand = 0.1,
                                                 recompute = TRUE,
                                                 colour = "gray40",
                                                 shadow = FALSE,
                                                 linewidth = 0.2,
                                                 scale.inset = c(1,0.3),
                                                 from = c(0.5, 4, 90, 99), 
                                                 to = c(0.38, 4.0 ,25, 70))

inset = line_main_s4003 + 
  theme(axis.ticks.length = unit(1, "pt"),
        axis.text.y = element_text(size = 4),
        axis.ticks.y = element_line(linewidth = 0.2, color = "black"))+
  inset_theme(axes = "y", margin=1)
line_cov_s4003 <- line_main_s4003 + geom_magnify(plot = inset,
                                                 expand = 0.1,
                                                 recompute = TRUE,
                                                 colour = "gray40",
                                                 shadow = FALSE,
                                                 linewidth = 0.2,
                                                 scale.inset = c(1,0.3),
                                                 from = c(0.5, 4, 90, 99), 
                                                 to = c(0.38, 4.0 ,25, 70))

# mse and cov
line_s2002 <- (line_mse_s2002 / line_cov_s2002) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

line_s2003 <- (line_mse_s2003 / line_cov_s2003) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

line_s2012 <- (line_mse_s2012 / line_cov_s2012) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

line_s2013 <- (line_mse_s2013 / line_cov_s2013) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

line_s3001 <- (line_mse_s3001 / line_cov_s3001) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

line_s3002 <- (line_mse_s3002 / line_cov_s3002) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

line_s3004 <- (line_mse_s3004 / line_cov_s3004) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

line_s3005 <- (line_mse_s3005 / line_cov_s3005) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

line_s4001 <- (line_mse_s4001 / line_cov_s4001) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

line_s4002 <- (line_mse_s4002 / line_cov_s4002) +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "left",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8))

plot_s2002 <- (box_s2002 / line_s2002) +  plot_layout(ncol = 1)
plot_s2003 <- (box_s2003 / line_s2003) +  plot_layout(ncol = 1)
plot_s2012 <- (box_s2012 / line_s2012) +  plot_layout(ncol = 1)
plot_s2013 <- (box_s2013 / line_s2013) +  plot_layout(ncol = 1)
plot_s3001 <- (box_s3001 / line_s3001) +  plot_layout(ncol = 1)
plot_s3002 <- (box_s3002 / line_s3002) +  plot_layout(ncol = 1)
plot_s3004 <- (box_s3004 / line_s3004) +  plot_layout(ncol = 1)
plot_s3005 <- (box_s3005 / line_s3005) +  plot_layout(ncol = 1)

plot_s4001 <- (box_s4001 / line_s4001) +  plot_layout(ncol = 1)
plot_s4002 <- (box_s4002 / line_s4002) +  plot_layout(ncol = 1)

plot_s4003 <- (box_s4003 / line_s4003) +  plot_layout(ncol = 1)
plot_s4004 <- (box_s4004 / line_s4004) +  plot_layout(ncol = 1)

# ABC
# 678*501
Scenario1 <- ggdraw() +
  draw_plot(plot_s2002) +
  draw_label("A", x = 0.18, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.18, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.59, y = 0.5, size = 10, color = "black")
Scenario2 <- ggdraw() +
  draw_plot(plot_s2003) +
  draw_label("A", x = 0.18, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.18, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.59, y = 0.5, size = 10, color = "black")
Scenario3 <- ggdraw() +
  draw_plot(plot_s2012) +
  draw_label("A", x = 0.18, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.18, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.59, y = 0.5, size = 10, color = "black")
Scenario4 <- ggdraw() +
  draw_plot(plot_s2013) +
  draw_label("A", x = 0.18, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.18, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.59, y = 0.5, size = 10, color = "black")
Scenario5 <- ggdraw() +
  draw_plot(plot_s4001) +
  draw_label("A", x = 0.18, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.18, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.59, y = 0.5, size = 10, color = "black")

Scenario7 <- ggdraw() +
  draw_plot(plot_s3001) +
  draw_label("A", x = 0.21, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.21, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.6, y = 0.5, size = 10, color = "black")
Scenario8 <- ggdraw() +
  draw_plot(plot_s3002) +
  draw_label("A", x = 0.21, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.21, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.6, y = 0.5, size = 10, color = "black")
Scenario9 <- ggdraw() +
  draw_plot(plot_s3004) +
  # draw_plot(box_zoom_s3004,x = 0.29, y = 0.76, width = 0.45, height = 0.14) +
  draw_label("A", x = 0.21, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.21, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.6, y = 0.5, size = 10, color = "black")
  # draw_line(c(0.3, 0.295), c(0.29, 0.31), color = "gray40", linewidth = 0.2, linetype = "dashed") +
  # draw_line(c(0.56, 0.4), c(0.29, 0.41), color = "gray40", linewidth = 0.2, linetype = "dashed")
Scenario10 <- ggdraw() +
  draw_plot(plot_s3005) +
  draw_label("A", x = 0.21, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.21, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.6, y = 0.5, size = 10, color = "black")
Scenario11 <- ggdraw() +
  draw_plot(plot_s4002) +
  draw_label("A", x = 0.21, y = 0.97, size = 10, color = "black") +
  draw_label("B", x = 0.21, y = 0.5, size = 10, color = "black") +
  draw_label("C", x = 0.6, y = 0.5, size = 10, color = "black")

