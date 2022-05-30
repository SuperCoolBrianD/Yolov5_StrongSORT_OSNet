# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:49:55 2022

@author: salma
"""

import numpy as np
from math import*
from scipy.stats.distributions import chi2
from scipy import linalg
from track import*


class track_MM:
    def __init__(self,z0,G_List,H,Q_List,R,maxVals,pInit,startSample,Ts,models,filters,sensor,N):
        r = len(models) #number of models
        
        sigma_x = sqrt(R[0,0])
        sigma_y = sqrt(R[1,1])
        
        maxVel = maxVals[0]
        maxAcc = maxVals[1]
        omegaMax = maxVals[2]
        
        #apply 1-point initialization
        xPost0 = np.array([z0[0],z0[1],0,0,0,0,0]) #initial state
        P_Post0 = np.diag(np.array([sigma_x**2,sigma_y**2,(maxVel/2)**2,(maxVel/2)**2,(maxAcc/2)**2,(maxAcc/2)**2,(omegaMax/2)**2])) #initial co-variance

        #xPostsList = [0]*r #mode-conditioned state estimates
        #P_PostsList = [0]*r #mode-conditioned state co-variances
        #pCurrents = [0]*r #mode-conditioned probabilitiy of track existences
        modeTrackList = [0]*r #list of mode-conditioned tracks
        
        isMM = True
        
        for i in range(r): #initiate mode-conditioned tracks
            G_i = G_List[i]
            Q_i = Q_List[i]
            modelType_i = models[i]
            
            track_i = track(z0, G_i, H, Q_i, R, maxVel,maxAcc, omegaMax, pInit, startSample, Ts, modelType_i, sensor, isMM, N)
            modeTrackList[i] = track_i 
            
        n = xPost0.shape[0] #dimension of state vector
        xEsts = np.zeros((n,N)) #array of state estimates initialized 
        
        xEsts[:,startSample] = xPost0 #store initial state at the starting sample/scan/frame
        ePost0 = z0 - H@xPost0 #a-posteriori measurement error required for SVSF
        
        #initiate mode probabilities
        if r==2:
            uVec0 = np.array([1/2,1/2])
        elif r==3:
            uVec0 = np.array([1/3,1/3,1/3])
        elif r==3:
            uVec0 = np.array([1/4,1/4,1/4,1/4])
            
        #Properties for track
        self.xPost = xPost0 #state estimate
        self.P_Post = P_Post0 #state co-variance
        self.ePost = ePost0 #a-posteriori error
        self.xEsts = xEsts #all state estimates
        self.modeTrackList = modeTrackList #mode matched tracks in list
        self.uVec = uVec0 #mode probability vector
        self.Ts = Ts #sampling time
        self.status = 1 #track status, 0 means deleted, 1 means tenative, and 2 means confirmed
        self.startSample = startSample #starting frame/scan/sample
        self.sensor = sensor #sensor for tracking
        self.models = models #models used in filter-bank
        self.filters = filters #filter applied in the filter-bank
        self.G_List = G_List #input gain matrices for each model
        self.Q_List = Q_List #process noise covariances for each model
        self.R = R #measurement noise co-variance
        self.H = H #measurement matrix
        self.isMM = True#if the tracker applies a multiple model filter
        self.endSample = None
        #self.pCurrent = pCurrents
        
    def IMMIPDAKF(self,measSet,MP,PD,PG,lambdaVal,maxVals,sensorPos):
        xPost = self.xPost
        modeTrackList = self.modeTrackList #list of mode-conditioned tracks
        uVec = self.uVec #mode probabilities
        modelList = self.models #models for IMM estimator
        gateArr = self.gateArr #binary array indicating which measurements have been associated to the track
        
        gateArr = gateArr.astype(int) #convert array values to integer
        ng= sum(gateArr) #number of measurements in the gate
        
        gatedMeas = measSet[:,gateArr.astype(bool)] #gated measurements
        
        
        velMax = maxVals[0] #used in 1-point initialization
        maxAcc = maxVals[1]
        omegaMax = maxVals[2]
        
        n = xPost.shape[0] #number of states
        
        r = len(modeTrackList) #number of models in IMMIPDA
        
        #Step 1) Computation of Mixing Probabilities
        cVec = (uVec.T@MP).T #normalization constants
        uMatMix = np.zeros((r,r))
        for i in range(r):
            for j in range(r):
                uMatMix[i,j]=(1/cVec[j])*MP[i,j]*uVec[i]    
                
                
        #Step 2) Mixing
        init_xPostsList = [0]*r #initial mode-matched state estimates
        init_P_PostsList = [0]*r #initial mode-matched co-variances
        new_pCurrentsInit = [0]*r #initial mode-matched probability of track existences
        for j in range(r):
            weightSum = np.zeros((n,))
            pNew = 0
            for i in range(r): #perform mixing
                x_i = modeTrackList[i].xPost #mode-conditioned state
                p_i = modeTrackList[i].pCurrent #mode-conditioned track probability
                weightSum = weightSum + x_i*uMatMix[i,j]
                pNew = pNew + p_i*uMatMix[i,j]
            
            init_xPostsList[j] = weightSum #compute initial mode-conditioned state estimate via mixing
            new_pCurrentsInit[j] = pNew
        
        for j in range(r):
            weightSumMat = np.zeros((n,n))
            x0_j = init_xPostsList[j]
            for i in range(r): #perform mixing
               x_i = modeTrackList[i].xPost
               P_i = modeTrackList[i].P_Post
                
               if modelList[j]== "CT" and (modelList[i] == "CV" or modelList[i]=="CA"):
                   weightSumMat[6,6] = (omegaMax/5)**2
               elif modelList[j]=="CA" and modelList[i] == "CV":
                   weightSumMat[4,4] = (maxAcc/30)**2
                   weightSumMat[5,5]= (maxAcc/30)**2
                    
               weightSumMat = weightSumMat + uMatMix[i,j]*(P_i + np.outer(x_i-x0_j, x_i-x0_j) )
                
            P0_j = weightSumMat #compute initial mode-conditioned co-variance via mixing
            init_P_PostsList[j] = P0_j
           
        #Step 3) Mode-Matched Filtering
        L = np.zeros((r,)) #likelihood functions
        
        gateArr = self.gateArr
        
        #Because there are multiple models, thus there are multiple gates, the union of the measurements is taken
        for i in range(r): #apply the PDA-KF using each model
            modeTrack_i = modeTrackList[i] #mode-conditioned track
        
            x0_i = init_xPostsList[i] #initialized state
            P0_i = init_P_PostsList[i] #initialized co-variance
            p_i = new_pCurrentsInit[i] #initizlied track existence probability
            
            
            #initialize track with new state estimate, co-variance, and track probability
            modeTrack_i.xPost = x0_i 
            modeTrack_i.P_Post = P0_i
            modeTrack_i.pCurrent = p_i
            modeTrack_i.gateArr = gateArr
            
            modeTrack_i.IPDAKF(measSet, MP, PD, PG, lambdaVal, sensorPos) #update state of mode-conditioned track
            zPred = modeTrack_i.zPred
            S = modeTrack_i.S
            
            modeTrackList[i] = modeTrack_i #update mode-conditioned track
        
            #Obtain likelihood function
            sumL= 0                
            
            abs_det = abs(np.linalg.det(2*pi*S))
            T = 1/(sqrt(abs_det))
            S_inv = np.linalg.inv(S)
            
            for j in range(ng):
                z_j = gatedMeas[:,j]
                nu_j = z_j-zPred
                
                sumL = sumL+ PD*T*exp(-.5*nu_j.T@S_inv@nu_j)

            #L[i] = (1-PD*PG)*(V**(-ng)) + ((PD*PG)/ng)*(V**(-ng+1))*sumL        
            L[i] = (1-PD*PG)*lambdaVal + sumL 
        
        #Step 4) Mode Probability Update
        c = np.inner(L,cVec)
        uVec = (1/c)*L*cVec
        
        #Step 5) Update state estimate, co-variance, and track probability for output purposes
        xEst = np.zeros((n,))
        pEst = 0
        
        for i in range(r):
            xPost_i = modeTrackList[i].xPost
            pCurrent_i = modeTrackList[i].pCurrent
            u_i = uVec[i]
            xEst = xEst + u_i*xPost_i
            pEst = pEst + u_i*pCurrent_i
            
        P_PostEst = np.zeros((n,n))
        for i in range(r):
            xPost_i = modeTrackList[i].xPost
            P_Post_i =modeTrackList[i].P_Post
            u_i = uVec[i]
            
            P_PostEst = P_PostEst + u_i*(P_Post_i+  np.outer(xPost_i-xEst,xPost_i-xEst))
            
        self.xPost = xEst
        self.P_Post = P_PostEst
        self.pCurrent = pEst
        self.uVec = uVec
        self.modeTrackList =modeTrackList     
        
        return xEst, pEst, uVec
        
    def IMMIPDA_Initialize(self,MP, maxVals):
        #Performs initialization step for the IMM estimator
        xPost = self.xPost
        modeTrackList = self.modeTrackList #list of mode-conditioned tracks
        uVec = self.uVec #mode probabilities
        modelList = self.models #models for IMM estimator
        
        #velMax = maxVals[0] #used in 1-point initialization
        maxAcc = maxVals[1]
        omegaMax = maxVals[2]
        
        n = xPost.shape[0] #number of states
        
        r = len(modeTrackList) #number of models in IMMIPDA
        
        #Step 1) Computation of Mixing Probabilities
        cVec = (uVec.T@MP).T #normalization constants
        uMatMix = np.zeros((r,r))
        for i in range(r):
            for j in range(r):
                uMatMix[i,j]=(1/cVec[j])*MP[i,j]*uVec[i]    
                
                
        #Step 2) Mixing
        #init_xPostsList = [0]*r #initial mode-matched state estimates
        #init_P_PostsList = [0]*r #initial mode-matched co-variances
        #new_pCurrentsInit = [0]*r #initial mode-matched probability of track existences
        for j in range(r):
            weightSum = np.zeros((n,))
            pNew = 0
            for i in range(r): #perform mixing
                x_i = modeTrackList[i].xPost #mode-conditioned state
                p_i = modeTrackList[i].pCurrent #mode-conditioned track probability
                weightSum = weightSum + x_i*uMatMix[i,j]
                pNew = pNew + p_i*uMatMix[i,j]
        
            modeTrackList[j].xPostNew = weightSum #store initial state
            modeTrackList[j].pCurrentNew = pNew #store initial track probability
            
            #init_xPostsList[j] = weightSum #compute initial mode-conditioned state estimate via mixing
            #new_pCurrentsInit[j] = pNew
        
        for j in range(r):
            weightSumMat = np.zeros((n,n))
            #x0_j = init_xPostsList[j]
            x0_j = modeTrackList[j].xPostNew 
            for i in range(r): #perform mixing
               x_i = modeTrackList[i].xPost
               P_i = modeTrackList[i].P_Post
                
               if modelList[j]== "CT" and (modelList[i] == "CV" or modelList[i]=="CA"):
                   weightSumMat[6,6] = (omegaMax)**2
               elif modelList[j]=="CA" and modelList[i] == "CV":
                   weightSumMat[4,4] = (maxAcc/25)**2
                   weightSumMat[5,5]= (maxAcc/25)**2
                    
               weightSumMat = weightSumMat + uMatMix[i,j]*(P_i + np.outer(x_i-x0_j, x_i-x0_j) )
                
            P0_j = weightSumMat #compute initial mode-conditioned co-variance via mixing
            modeTrackList[j].P_PostNew = P0_j #store initial co-variance
            #init_P_PostsList[j] = P0_j
            
        self.modeTrackList = modeTrackList
        self.cVec = cVec
           
    def IMMIPDA_Filter(self,measSet,MP,PD,PG,lambdaVal,maxVals,sensorPos):
        #Performs the steps required for state estimation, steps 1-3
        xPost = self.xPost
        modeTrackList = self.modeTrackList #list of mode-conditioned tracks
        uVec = self.uVec #mode probabilities
        cVec = self.cVec #cVec from initialization
        modelList = self.models #models for IMM estimator
        gateArr = self.gateArr #binary array indicating which measurements have been associated to the track
        
        gateArr = gateArr.astype(int) #convert array values to integer
        ng= sum(gateArr) #number of measurements in the gate
        
        gatedMeas = measSet[:,gateArr.astype(bool)] #gated measurements
        
        #Step 3) Mode-Matched Filtering
        #cVec = (uVec.T@MP).T #normalization constants
                
        n = xPost.shape[0] #number of states
        r = len(modelList) #number of models
        
        L = np.zeros((r,)) #likelihood functions
        #Because there are multiple models, thus there are multiple gates, the union of the measurements is taken
        for i in range(r): #apply the PDA-KF using each model
            modeTrack_i = modeTrackList[i] #mode-conditioned track
            modeTrack_i.gateArr = gateArr #stores the union 
            
            modeTrack_i.IPDAKF(measSet, MP, PD, PG, lambdaVal, sensorPos) #update state of mode-conditioned track
            zPred = modeTrack_i.zPred
            S = modeTrack_i.S
            
            modeTrackList[i] = modeTrack_i #update mode-conditioned track
        
            #Obtain likelihood function
            sumL= 0                
            
            abs_det = abs(np.linalg.det(2*pi*S))
            T = 1/(sqrt(abs_det))
            S_inv = np.linalg.pinv(S)
            
            for j in range(ng):
                z_j = gatedMeas[:,j]
                nu_j = z_j-zPred
                
                sumL = sumL+ PD*T*exp(-.5*nu_j.T@S_inv@nu_j)

            #L[i] = (1-PD*PG)*(V**(-ng)) + ((PD*PG)/ng)*(V**(-ng+1))*sumL        
            L[i] = (1-PD*PG)*lambdaVal + sumL 
        
        #Step 4) Mode Probability Update
        c = np.inner(L,cVec)
        uVec = (1/c)*L*cVec
        
        #Step 5) Update state estimate, co-variance, and track probability for output purposes
        xEst = np.zeros((n,))
        pEst = 0
        
        for i in range(r):
            xPost_i = modeTrackList[i].xPost
            pCurrent_i = modeTrackList[i].pCurrent
            u_i = uVec[i]
            xEst = xEst + u_i*xPost_i
            pEst = pEst + u_i*pCurrent_i
            
        P_PostEst = np.zeros((n,n))
        for i in range(r):
            xPost_i = modeTrackList[i].xPost
            P_Post_i =modeTrackList[i].P_Post
            u_i = uVec[i]
            
            P_PostEst = P_PostEst + u_i*(P_Post_i+  np.outer(xPost_i-xEst,xPost_i-xEst))
            
        self.xPost = xEst
        self.P_Post = P_PostEst
        self.pCurrent = pEst
        self.uVec = uVec
        self.modeTrackList =modeTrackList     
        
        return xEst, pEst, uVec
        
        

    def stackState(self,k): 
        #stores the state estimate from time step k
        xEsts = self.xEsts #array of state estimates
        xEsts[:,k] = self.xPost #include state estimate in the array
        
    def stackBL(self,k):
        BLs = self.BLs
        BLs[:,k] = self.BL_Vec
        
        self.BLs = BLs
        
        
'''
def gating(xPost,P_Post,G,H,Q,R,PG,sensorPos,measSet,Ts,modelType,useRadar):
    n = int(xPost.shape[0]) #number of dimensions of the state
    m = measSet.shape[0]
    
    gamma = chi2.ppf(PG, df=m)

    #Predict step
    if modelType=="CV":
        xPred,F_k = CVModel(xPost,np.diag(np.array([0,0,0])),Ts)
    elif modelType=="CT":
        xPred,F_k,LinF = CTModel(xPost, np.diag(np.array([0,0,0])), Ts)    
        F_k = LinF
    
    if Q.shape[0]>3:
        P_Pred = F_k@P_Post@F_k.T + Q
    else:
        P_Pred = F_k@P_Post@F_k.T + G@Q@G.T
    
    P_Pred = 0.5*(P_Pred + P_Pred.T)
    
    if useRadar==True:
        rangePred = getRange(xPred,sensorPos) #predict measurement using non-linear model
        anglePred = getAngle(xPred,sensorPos)
        
        m=2
        zPred = np.zeros((m,))
        zPred[0] = rangePred #store into a vector
        zPred[1] = anglePred
        
        H = obtain_Jacobian_H(xPred, sensorPos) #obtain Jacobian matrix
    else:
        zPred = H@xPred
        
    S = H@P_Pred@H.T + R
    S_inv = np.linalg.inv(S)
    
    numMeas = measSet.shape[1]
    
    gateArr = np.zeros((numMeas,))
    for i in range(numMeas):
        z_i = measSet[:,i]
        nu_i = z_i-zPred
        dist = nu_i.T @ S_inv @nu_i
        
        if dist<=gamma:
            gateArr[i] = 1
            
    return gateArr        
'''
        
        
def PDAKF(xPost,P_Post,G,H,Q,R,Ts,sensorPos,measSet,gatedArr,PG,PD,lambdaVal,modelType,useRadar):
    
    #Predict step
    if modelType=="CV":
        xPred,F_k = CVModel(xPost,np.diag(np.array([0,0,0])),Ts)
    elif modelType=="CT":
        xPred,F_k,LinF = CTModel(xPost, np.diag(np.array([0,0,0])), Ts)    
        F_k = LinF
    
    if Q.shape[0]>3:
        P_Pred = F_k@P_Post@F_k.T + Q
    else:
        P_Pred = F_k@P_Post@F_k.T + G@Q@G.T
    
    P_Pred = 0.5*(P_Pred + P_Pred.T)
    
    if useRadar==True:
        rangePred = getRange(xPred,sensorPos) #predict measurement using non-linear model
        anglePred = getAngle(xPred,sensorPos)
        
        m=2
        zPred = np.zeros((m,))
        zPred[0] = rangePred #store into a vector
        zPred[1] = anglePred
        
        H = obtain_Jacobian_H(xPred, sensorPos) #obtain Jacobian matrix
    else:
        zPred = H@xPred
        
    S = H@P_Pred@H.T + R
    S_inv = np.linalg.inv(S)
    K = P_Pred@H.T@ S_inv
    
    m = zPred.shape[0]
    
    numGated = int(sum(gatedArr))
    
    xMeas = measSet[0,:]
    yMeas = measSet[1,:]
    gatedMeas = np.vstack((xMeas[gatedArr==1],yMeas[gatedArr==1]))
    
    L_Vals = np.zeros(numGated,)
    innovations = np.zeros((m,numGated))
    nu_k = np.zeros(m,)
    
    abs_det = abs(np.linalg.det(2*pi*S))
    #T =   1/(  ((2*pi)**(m/2)) *sqrt(abs_det_S))
    T = 1/(sqrt(abs_det))
    for i in range(numGated):
        z_i = gatedMeas[:,i]
        nu_i = z_i-zPred
        innovations[:,i] = nu_i
        
        G_i = T*exp(-.5*nu_i.T@S_inv@nu_i)
        L_i =(G_i*PD)/lambdaVal
        L_Vals[i] = L_i
        
    sumL = sum(L_Vals)
    
    denomTerm= (1-PD*PG+sumL)
    
    Beta0 = (1-PD*PG)/denomTerm
    middleTerm = np.zeros((m,m))
    for i in range(numGated):
        L_i = L_Vals[i]
        nu_i=innovations[:,i]
        Beta_i = L_i/denomTerm
        nu_k = nu_k + Beta_i*nu_i
        
        middleTerm = middleTerm + Beta_i*np.outer(nu_i, nu_i)
        
    P_Tilda = K@(middleTerm - np.outer(nu_k,nu_k))@K.T
    P_c = P_Pred - K@S@K.T
    P_PostNew = Beta0*P_Pred + (1-Beta0)*P_c + P_Tilda
    P_PostNew = 0.5*(P_PostNew + P_PostNew.T)
    
    xPostNew = xPred + K@nu_k    
    
    return xPostNew,P_PostNew,zPred,S   

def CVModel(xVec,Q,Ts):
    F_k = np.array([[1,0,Ts,0,0,0,0],
                    [0,1,0,Ts,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0]])
    G_k = np.array([[(Ts**2)/2, 0,0],
                    [0, (Ts**2)/2,0],
                    [Ts,0,0],
                    [0,Ts,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]])
    
    if Q.shape[0] > 3:
        mean = np.array([0,0,0,0,0])
        v = np.random.multivariate_normal(mean,Q) 
        xNew = F_k@xVec + v
    else:
         mean = np.array([0,0,0])
         v = np.random.multivariate_normal(mean,Q)   
         xNew = F_k@xVec + G_k@v    
    
    return xNew, F_k

def CTModel(xVec,Q,Ts):
    n = int(xVec.shape[0])
    vx_k = xVec[2]
    vy_k = xVec[3]
    omega_k = xVec[6]
    
    G_k = np.array([[(Ts**2)/2, 0,0],
                    [0, (Ts**2)/2,0],
                    [Ts,0,0],
                    [0,Ts,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,Ts]])

    LinF = np.zeros((n,n))
    
    if omega_k != 0:
        F_k = np.array([[1,0,sin(omega_k*Ts)/omega_k,(cos(omega_k*Ts)-1)/omega_k,0,0,0],
                       [0,1,(1-cos(omega_k*Ts))/omega_k,sin(omega_k*Ts)/omega_k,0,0,0],
                       [0,0, cos(omega_k*Ts), -sin(omega_k*Ts),0,0,0],
                       [0,0, sin(omega_k*Ts), cos(omega_k*Ts),0,0,0],
                       [0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,1]])
        
        LinF[0,0] = 1
        LinF[0,2] = sin(Ts*omega_k)/omega_k
        LinF[0,3] = (cos(Ts*omega_k)-1)/omega_k
        LinF[0,6] = (Ts*vx_k*cos(Ts*omega_k))/omega_k - (vy_k*(cos(Ts*omega_k) - 1))/omega_k**2 - (vx_k*sin(Ts*omega_k))/omega_k**2 - (Ts*vy_k*sin(Ts*omega_k))/omega_k;
        
        LinF[1,1] = 1
        LinF[1,2] =  -(cos(Ts*omega_k) - 1)/omega_k
        LinF[1,3] = sin(Ts*omega_k)/omega_k
        LinF[1,6]  =  (vx_k*(cos(Ts*omega_k) - 1))/omega_k**2 - (vy_k*sin(Ts*omega_k))/omega_k**2 + (Ts*vy_k*cos(Ts*omega_k))/omega_k + (Ts*vx_k*sin(Ts*omega_k))/omega_k;
        
        LinF[2,2] = cos(Ts*omega_k)
        LinF[2,3] = -sin(Ts*omega_k)
        LinF[2,6] = - Ts*vy_k*cos(Ts*omega_k) - Ts*vx_k*sin(Ts*omega_k);
        
        LinF[3,2] = sin(Ts*omega_k)
        LinF[3,3] = cos(Ts*omega_k)
        LinF[3,6] = Ts*vx_k*cos(Ts*omega_k) - Ts*vy_k*sin(Ts*omega_k);
       
        LinF[6,6] = 1
    else:
        F_k = np.array([[1,0,Ts,0,0,0,0],
                        [0,1,0,Ts,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1]])
        n = F_k.shape[0]
        LinF[0,0]=1
        LinF[0,2]=Ts
        LinF[0,6]=-(Ts**2*vy_k)/2
        
        LinF[1,1]=1
        LinF[1,3]=Ts
        LinF[1,6]=(Ts**2*vx_k)/2;
        
        LinF[2,2]=1
        LinF[2,6]=-Ts*vy_k
        
        LinF[3,3]=1
        LinF[3,6]=Ts*vx_k
        
        LinF[6,6]=1
        
    if Q.shape[0] > 3:
        mean = np.array([0,0,0,0,0])
        v = np.random.multivariate_normal(mean,Q) 
        xNew = F_k@xVec + v
    else:
         mean = np.array([0,0,0])
         v = np.random.multivariate_normal(mean,Q)   
         xNew = F_k@xVec + G_k@v
        
    return xNew, F_k, LinF


def CAModel(xVec,Q,Ts):
    F_k = np.array([[1,0,Ts,0,(1/2)*(Ts**2),0,0],
                    [0,1,0,Ts,0,(1/2)*(Ts**2),0],
                    [0,0,1,0,Ts,0,0],
                    [0,0,0,1,0,Ts,0],
                    [0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0]])
    
    G_k = np.array([[(Ts**2)/2, 0,0],
                    [0, (Ts**2)/2,0],
                    [Ts,0,0],
                    [0,Ts,0],
                    [1,0,0],
                    [0,1,0],
                    [0,0,0]])
    
    if Q.shape[0] > 3:
        mean = np.array([0,0,0,0,0])
        v = np.random.multivariate_normal(mean,Q) 
        xNew = F_k@xVec + v
    else:
         mean = np.array([0,0,0])
         v = np.random.multivariate_normal(mean,Q)   
         xNew = F_k@xVec + G_k@v    
    
    return xNew, F_k