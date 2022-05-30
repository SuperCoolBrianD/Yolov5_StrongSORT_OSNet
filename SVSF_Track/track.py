# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 21:00:06 2022

@author: salma
"""
import numpy as np
import math
from scipy.stats.distributions import chi2
from scipy import linalg


class track:
    def __init__(self,z0,G,H,Q,R,maxVel, maxAcc,omegaMax,pInit,startSample,Ts, modelType, sensor,isMM, N):
        #initializes track using 1-point initialization
        
        #Inputs: 
            #z0 = measurement to initiate track
            #G = input gain matrix
            #H = measurement matrix
            #Q = process noise co-variance
            #R = measurement noise co-variance
            #sigma_w = measurement standard deviation 
            #maxVel = maximum velocity
            #omegaMax = maximum turn-rate
            #pInit = initial probability of track existence
            #status = initial status
            #startSample = starting frame/scan/sample for the track
            #Ts = sampling time
            #modelType = model for the filter
            #sensor = sensor used to get measurements
            #N = total number of samples in the simulation
            
        
        #apply 1-point initialization
        xPost0 = np.array([z0[0],z0[1],0,0,0,0,0]) #initial state
        
        sigma_x = math.sqrt(R[0,0])
        sigma_y = math.sqrt(R[1,1])
        
        
        P_Post0 = np.diag(np.array([sigma_x**2,sigma_y**2,(maxVel/2)**2,(maxVel/2)**2,(maxAcc/2)**2,(maxAcc/2)**2,(omegaMax/2)**2])) #initial co-variance
        
        #properties of track
        self.xPost = xPost0
        self.P_Post = P_Post0
        self.status = 1
        self.startSample = startSample
        self.endSample = None
        self.sensor = sensor
        self.isMM = isMM
        self.Ts =  Ts
        self.modelType = modelType
        self.Q = Q
        self.R = R
        self.H = H
        self.G = G
        self.pCurrent = pInit
        
        n = xPost0.shape[0] #dimension of state vector
        xEsts = np.zeros((n,N)) #array of state estimates initialized 
        
        BLs = np.zeros((n,N)) #BL widths
        
        xEsts[:,startSample] = xPost0 #store initial state at the starting sample/scan/frame
        ePost0 = z0 - H@xPost0 #a-posteriori measurement error required for SVSF
        
        self.ePost = ePost0 #set ePost as a property
        self.xEsts = xEsts #estimated states is also a property
        self.BLs = BLs
                #uses the KF to update the state of a track
        
        #Inputs:
            #z_k = measurement at time step k
            #sensorPos = sensor position if radar is being used, if no radar is being used it can be anything
        
        #Outputs:
            #xPostNew = updated state estimate
            #P_PostNew = updated co-variance
        
        
    def PDAKF(self,measSet,lambdaVal,PG,PD,sensorPos):
        #Performs data association and filtering
        
        #Inputs: measSet = array of gated measurements
        #       lambdaVal = parameter for clutter distribution
        #       PG = gate probability
        #       PD = probability of detection
        #       sensorPos = sensor position if using radar
        
        #outputs: xPostNew = updated state
        #         P_PostNew = updated co-variance
        
        
        xPost = self.xPost #latest state estimate
        P_Post = self.P_Post #latest co-variance
        modelType = self.modelType #model for filter
        sensor = self.sensor #sensor type
        Ts = self.Ts #sampling time
        Q = self.Q #process noise co-variance
        R = self.R #measurement co-variance
        G = self.G #input gain matrix
        H = self.H #measurement matrix
        gatedArr = self.gateArr #binary array, if element i is 1 then that indicates that measurement i is in the gate of this track
        
        n = int(xPost.shape[0]) #number of dimensions of the state
        
        #Predict step
        if modelType=="CV":
            xPred,F_k = CVModel(xPost,np.diag(np.array([0,0,0])),Ts) #predict using the CV model
        elif modelType=="CA":
            xPred,F_k = CAModel(xPost, np.diag(np.array([0,0,0])), Ts)
        elif modelType=="CT":
            xPred,F_k,LinF = CTModel(xPost, np.diag(np.array([0,0,0])), Ts) #predict using the CT model
            F_k = LinF
        
        #predict co-variance
        if Q.shape[0]>3:
            P_Pred = F_k@P_Post@F_k.T + Q
        else:
            P_Pred = F_k@P_Post@F_k.T + G@Q@G.T
        
        P_Pred = 0.5*(P_Pred + P_Pred.T) #for numerical stability
        
        
        if sensor=="Radar": #if the sensor is radar
            rangePred = getRange(xPred,sensorPos) #predict measurement using non-linear model
            anglePred = getAngle(xPred,sensorPos)
            
            m=2
            zPred = np.zeros((m,))
            zPred[0] = rangePred #store into a vector
            zPred[1] = anglePred
            
            H = obtain_Jacobian_H(xPred, sensorPos) #obtain Jacobian matrix
        else:
            zPred = H@xPred #if the sensor is lidar use the linear measurement model
            
        S = H@P_Pred@H.T + R #innovation co-variance
        S_inv = np.linalg.inv(S)
        K = P_Pred@H.T@ S_inv #KF gain
        
        m = zPred.shape[0] #dimension of measurement vector
        
        numGated = int(sum(gatedArr)) #number of measurements in the gate of this track
        
        xMeas = measSet[0,:] #x positions of measurements
        yMeas = measSet[1,:]  #y positions of measurements
        gatedMeas = np.vstack((xMeas[gatedArr==1],yMeas[gatedArr==1])) #stack the x and y measurements
        
        L_Vals = np.zeros(numGated,) #likelihood rations array
        innovations = np.zeros((m,numGated)) #array of innovation vectors
        nu_k = np.zeros(m,) #weighted innovation vector
        
        abs_det = abs(np.linalg.det(2*math.pi*S)) #part of the likelihood formula
        #T =   1/(  ((2*math.pi)**(m/2)) *math.sqrt(abs_det_S))
        T = 1/(math.sqrt(abs_det)) #part of the likelihood ratio formula
        for i in range(numGated):
            z_i = gatedMeas[:,i]
            nu_i = z_i-zPred
            innovations[:,i] = nu_i #ith innovation
            
            G_i = T*math.exp(-.5*nu_i.T@S_inv@nu_i) #Gaussian probability density function
            L_i =(G_i*PD)/lambdaVal #ith likelihood ratio
            L_Vals[i] = L_i
            
        sumL = sum(L_Vals) #sum of liklihood ratios
        
        denomTerm= (1-PD*PG+sumL) #denominator term for obtaining association probabilities
        
        Beta0 = (1-PD*PG)/denomTerm #probability that no measurement belongs to the target
        middleTerm = np.zeros((m,m)) #part of the co-variance update term
        
        for i in range(numGated): #for each gated measurement
            L_i = L_Vals[i] #obtain likelihood ration
            nu_i=innovations[:,i] #obtain innovation vector
            Beta_i = L_i/denomTerm #obtain association probability that measurement i belongs to the target
            nu_k = nu_k + Beta_i*nu_i #weighted innovation co-variance
            
            middleTerm = middleTerm + Beta_i*np.outer(nu_i, nu_i) #term used in part of the co-variance update
            
        P_Tilda = K@(middleTerm - np.outer(nu_k,nu_k))@K.T #co-variance due to measurement origin uncertainty
        P_c = P_Pred - K@S@K.T #co-variance from the standard KF
        P_PostNew = Beta0*P_Pred + (1-Beta0)*P_c + P_Tilda #updated co-variance
        P_PostNew = 0.5*(P_PostNew + P_PostNew.T) #for numerical stability
        
        xPostNew = xPred + K@nu_k #updated state
        
        #store updated state and co-variance into the track object
        self.xPost = xPostNew
        self.P_Post = P_PostNew
        
        return xPostNew,P_PostNew
    
    def IPDAKF(self,measSet,MP,PD,PG,lambdaVal,sensorPos):
        #performs data association, filtering, and updates probability of track existence
        
        #Inputs: measSet = array of gated measurements
        #       lambdaVal = parameter for clutter distribution
        #       PG = gate probability
        #       PD = probability of detection
        #       sensorPos = sensor position if using radar
        
        #outputs: xPostNew = updated state
        #         P_PostNew = updated co-variance
        #         pCurrent = updated probability of track existence
        
        isMM = self.isMM
        
        if isMM==True: #obtain info from IMM initialization
            xPost = self.xPostNew #latest state estimate
            P_Post = self.P_PostNew #latest co-variance
            pCurrent = self.pCurrentNew #latest track existence probability
        else:
             xPost = self.xPost #latest state estimate
             P_Post = self.P_Post #latest co-variance
             pCurrent = self.pCurrent #latest track existence probability   
        
        modelType = self.modelType #filter model
        sensor = self.sensor #type of sensor
        Ts = self.Ts #sampling time
        Q = self.Q #process noise co-variance
        R = self.R #measurement noise co-variance
        G = self.G #input gain matrix
        H = self.H #measurement matrix
        gatedArr = self.gateArr #binary array, if element i is 1, that means measurement i lies in the gate of this track 
        
        n = xPost.shape[0] #dimension of state vector
        m = measSet.shape[0] #dimension of measurement matrix
        m_k = int(sum(gatedArr)) #number of gated measurements
        gamma = chi2.ppf(PG, df=m) #gating threshold from PG
        
        #Predict step
        if modelType=="CV":
            xPred,F_k = CVModel(xPost,np.diag(np.array([0,0,0])),Ts) #predict using CV model
        elif modelType=="CA":
            xPred,F_k = CAModel(xPost, np.diag(np.array([0,0,0])), Ts) #predict using CA model
        elif modelType=="CT":
            xPred,F_k,LinF = CTModel(xPost, np.diag(np.array([0,0,0])), Ts) #predict using CT model
            F_k = LinF
        
        #predict co-variance
        if Q.shape[0]>3:
            P_Pred = F_k@P_Post@F_k.T + Q
        else:
            P_Pred = F_k@P_Post@F_k.T + G@Q@G.T
        
        P_Pred = 0.5*(P_Pred + P_Pred.T) #for numerical stability 
        
        pVec = np.array([pCurrent,1-pCurrent]) 
        pPredVec = MP.T@pVec#predict track existence probability
        pPred = pPredVec[0] #probability of track existence, other element in pPred is the probability of no track existence
        
        if sensor=="Radar": 
            rangePred = getRange(xPred,sensorPos) #predict measurement using non-linear model
            anglePred = getAngle(xPred,sensorPos)
            
            m=2
            zPred = np.zeros((m,))
            zPred[0] = rangePred #store into a vector
            zPred[1] = anglePred
            
            H = obtain_Jacobian_H(xPred, sensorPos) #obtain Jacobian matrix
        else: #if LIDAR
            zPred = H@xPred #predict measurement
            
        S = H@P_Pred@H.T + R #innovation co-variance
        S = .5*(S + S.T)
        S_inv = np.linalg.pinv(S) #innovation co-variance inverse
        K = P_Pred@H.T@ S_inv #KF gain
        
        cn = math.pi #term used to get the volume of the gated region ellipsoid
        V_k = cn*math.sqrt(abs(np.linalg.det(gamma*S))) #volume of gated region
        
        if lambdaVal != 0: #if lambda is non-zero, compute mHat as follows
            mHat = lambdaVal*V_k
        else:
            mHat = 1- PD*PG*pPred #otherwise calculate mHat using this formula
        
        xMeas = measSet[0,:] #x component of measurements in gate
        yMeas = measSet[1,:] #y component of measurements in gate
        gateMeas = np.vstack((xMeas[gatedArr==1],yMeas[gatedArr==1])) #measurements in the gate stacked
        
        
        abs_det = abs(np.linalg.det(2*math.pi*S)) #term used to get the likelihood ratio 
        #T =   1/(  ((2*math.pi)**(m/2)) *math.sqrt(abs_det_S))
        T = 1/(math.sqrt(abs_det)) #term used to get the likelihood ratio
        innovations = np.zeros((m,m_k)) #innovation vectors
        
        prob_z_i_Vals = np.zeros((m_k,)) #likelihood function of measurements
        for i in range(m_k):
            z_i = gateMeas[:,i] #ith gated measurement
            nu_i = z_i-zPred #innovation
            innovations[:,i] = nu_i
            
            G_i = T*math.exp(-.5*nu_i.T@S_inv@nu_i) #likelihood function of measurement
            
            prob_z_i = G_i/PG #probability of measurement
            prob_z_i_Vals[i]=prob_z_i
          #  probZ_Sum = probZ_Sum+ prob_z_i
        probZ_Sum = sum(prob_z_i_Vals) #sum of probabilities 

        #print(mHat)
        if m_k==0:
            delta_k = PD*PG #if m_k is zero use this formula to get the delta_k term
        else:
            delta_k = PD*PG- PD*PG*(V_k/mHat)*probZ_Sum #otherwise, apply this formula to obtain delta_k
            
        pCurrent = ((1-delta_k)/(1-delta_k*pPred))*pPred #update probability of track existence
        
        Beta0 = (1-PD*PG)/(1-delta_k) #probability of no measurement belonging to a target
        BetaVals = np.zeros((m_k,)) #association probabilities for measurements
        #compute association probabilities
        for i in range(m_k):
            prob_z_i = prob_z_i_Vals[i] #probability of z_i from earlier
            Beta_i = (PD*PG*(V_k/mHat)*prob_z_i)/(1-delta_k) #association probability of z_i belonging to the target
            BetaVals[i] = Beta_i
            
        nu_k = np.zeros((m,)) #weighted innovation
        middleTerm = np.zeros((m,m)) #term part of the co-variance update formula
        for i in range(m_k): #for each gated measurement
            nu_i=innovations[:,i] #ith innovation
            Beta_i = BetaVals[i] #association probability
            nu_k = nu_k + Beta_i*nu_i #weighted innovation
            
            middleTerm = middleTerm + Beta_i*np.outer(nu_i, nu_i) #term used to get a co-variance term
            
        P_Tilda = K@(middleTerm - np.outer(nu_k,nu_k))@K.T #co-variance due to measurement origin uncertainty
        P_c = P_Pred - K@S@K.T #co-variance from standard KF
        P_PostNew = Beta0*P_Pred + (1-Beta0)*P_c + P_Tilda #updated state co-variance
        P_PostNew = 0.5*(P_PostNew + P_PostNew.T) #for numerical stability
        
        xPostNew = xPred + K@nu_k    #updated state estimate
        
        #store updated state estimate, updated co-variance, and updated probability of track existence
        self.xPost = xPostNew
        self.P_Post = P_PostNew
        self.pCurrent = pCurrent
        self.zPred = zPred
        self.S = S
        self.V = V_k
        
        return xPostNew,P_PostNew
    
    def IPDAGVBLSVSF(self,measSet,MP,PD,PG,lambdaVal,sensorPos, T_mat, gammaZ, gammaY, psiZ, psiY):
        #performs data association, filtering, and updates probability of track existence
        
        #Inputs: measSet = array of gated measurements
        #       lambdaVal = parameter for clutter distribution
        #       PG = gate probability
        #       PD = probability of detection
        #       sensorPos = sensor position if using radar
        
        #outputs: xPostNew = updated state
        #         P_PostNew = updated co-variance
        #         pCurrent = updated probability of track existence
        
        xPost = self.xPost #latest state estimate
        P_Post = self.P_Post #latest co-variance
        ePost = self.ePost #latest a-posteriori measurement error
        pCurrent = self.pCurrent #latest track existence probability
        modelType = self.modelType #filter model
        sensor = self.sensor #type of sensor
        Ts = self.Ts #sampling time
        Q = self.Q #process noise co-variance
        R = self.R #measurement noise co-variance
        G = self.G #input gain matrix
        H = self.H #measurement matrix
        gatedArr = self.gateArr #binary array, if element i is 1, that means measurement i lies in the gate of this track 
        BLs = self.BLs
        
        n = xPost.shape[0] #dimension of state vector
        m = measSet.shape[0] #dimension of measurement matrix
        m_k = int(sum(gatedArr)) #number of gated measurements
        gamma = chi2.ppf(PG, df=m) #gating threshold from PG
        
        #Predict step
        if modelType=="CV":
            xPred,F_k = CVModel(xPost,np.diag(np.array([0,0,0])),Ts) #predict using CV model
        elif modelType == "CA":
            xPred,F_k = CAModel(xPost,np.diag(np.array([0,0,0])),Ts) #predict using CV model
        elif modelType=="CT":
            xPred,F_k,LinF = CTModel(xPost, np.diag(np.array([0,0,0])), Ts) #predict using CT model
            F_k = LinF
        
        #predict co-variance
        if Q.shape[0]>3:
            P_Pred = F_k@P_Post@F_k.T + Q
        else:
            P_Pred = F_k@P_Post@F_k.T + G@Q@G.T
        
        P_Pred = 0.5*(P_Pred + P_Pred.T) #for numerical stability 
        
        pVec = np.array([pCurrent,1-pCurrent]) 
        pPredVec = MP.T@pVec#predict track existence probability
        pPred = pPredVec[0] #probability of track existence, other element in pPred is the probability of no track existence
        
        if sensor=="Radar": 
            rangePred = getRange(xPred,sensorPos) #predict measurement using non-linear model
            anglePred = getAngle(xPred,sensorPos)
            
            m=2
            zPred = np.zeros((m,))
            zPred[0] = rangePred #store into a vector
            zPred[1] = anglePred
            
            H = obtain_Jacobian_H(xPred, sensorPos) #obtain Jacobian matrix
        else: #if LIDAR
            zPred = H@xPred #predict measurement
            
        S = H@P_Pred@H.T + R #innovation co-variance
        S = 0.5*(S+S.T)
        S_inv = np.linalg.inv(S) #innovation co-variance inverse
                
        cn = math.pi #term used to get the volume of the gated region ellipsoid
        V_k = cn*math.sqrt(abs(np.linalg.det(gamma*S))) #volume of gated region
        
        if lambdaVal != 0: #if lambda is non-zero, compute mHat as follows
            mHat = lambdaVal*V_k
        else:
            mHat = 1- PD*PG*pPred #otherwise calculate mHat using this formula
        
        xMeas = measSet[0,:] #x component of measurements in gate
        yMeas = measSet[1,:] #y component of measurements in gate
        gateMeas = np.vstack((xMeas[gatedArr==1],yMeas[gatedArr==1])) #measurements in the gate stacked
        
        abs_det = abs(np.linalg.det(2*math.pi*S)) #term used to get the likelihood ratio 
        #T =   1/(  ((2*math.pi)**(m/2)) *math.sqrt(abs_det_S))
        T = 1/(math.sqrt(abs_det)) #term used to get the likelihood ratio
        innovations = np.zeros((m,m_k)) #innovation vectors
        
        prob_z_i_Vals = np.zeros((m_k,)) #likelihood function of measurements
        for i in range(m_k):
            z_i = gateMeas[:,i] #ith gated measurement
            nu_i = z_i-zPred #innovation
            innovations[:,i] = nu_i
            
            G_i = T*math.exp(-.5*nu_i.T@S_inv@nu_i) #likelihood function of measurement
            
            prob_z_i = G_i/PG #probability of measurement
            prob_z_i_Vals[i]=prob_z_i
          #  probZ_Sum = probZ_Sum+ prob_z_i
        probZ_Sum = sum(prob_z_i_Vals) #sum of probabilities 
        
        if m_k==0:
            delta_k = PD*PG #if m_k is zero use this formula to get the delta_k term
        else:
            delta_k = PD*PG- PD*PG*(V_k/mHat)*probZ_Sum #otherwise, apply this formula to obtain delta_k
            
        pCurrent = ((1-delta_k)/(1-delta_k*pPred))*pPred #update probability of track existence
        
        Beta0 = (1-PD*PG)/(1-delta_k) #probability of no measurement belonging to a target
        BetaVals = np.zeros((m_k,)) #association probabilities for measurements
        #compute association probabilities
        for i in range(m_k):
            prob_z_i = prob_z_i_Vals[i] #probability of z_i from earlier
            Beta_i = (PD*PG*(V_k/mHat)*prob_z_i)/(1-delta_k) #association probability of z_i belonging to the target
            BetaVals[i] = Beta_i
            
        nu_k = np.zeros((m,)) #weighted innovation
        middleTerm = np.zeros((m,m)) #term part of the co-variance update formula
        for i in range(m_k): #for each gated measurement
            nu_i=innovations[:,i] #ith innovation
            Beta_i = BetaVals[i] #association probability
            nu_k = nu_k + Beta_i*nu_i #weighted innovation
            
            middleTerm = middleTerm + Beta_i*np.outer(nu_i, nu_i) #term used to get a co-variance term
            
        H_1 = H[0:m,0:m] #sub-matrix of measurement matrix
        
        ePred = nu_k #a-priori measurement error set to combined innovation
        ePred[ePred==0] = 1E-1000
        
        Psi_k = T_mat@F_k @linalg.inv(T_mat) #perform transformation

        #extract sub-matrices
        Psi_12 = Psi_k[0:m,m:n]
        Psi_22 = Psi_k[m:n,m:n]
        
        P_Pred_11 = P_Pred[0:m,0:m]
        P_Pred_21 = P_Pred[m:n,0:m]
        
        E_z = abs(ePred) + gammaZ@abs(ePost)
        EzBar = np.diag(E_z)
        psiZ_opt = linalg.pinv(linalg.pinv(EzBar)@H_1 @P_Pred_11 @H_1.T @S_inv)     #optimal smoothing boundary layer widths for measured states

        
        if m!=n: #if using the m<n optimal SVSF
            pinv_psi_12 = linalg.pinv(Psi_12)
            E_y = abs(Psi_22 @ pinv_psi_12 @ ePred) + gammaY@abs(pinv_psi_12 @ePred)
            EyBar = np.diag(E_y)
            M = Psi_22 @ pinv_psi_12
            psiY_opt = linalg.pinv(linalg.pinv(EyBar)@P_Pred_21@H_1.T@S_inv @linalg.pinv(M)) #optimal smoothing boundary layer widths for unmeasured states
        
            if psiZ_opt[0,0] < psiZ[0] and psiZ_opt[1,1] < psiZ[1]:
                Ku = P_Pred_11@H_1.T@S_inv#KF upper gain
            else:
                Ku = linalg.inv(H_1)@ np.diag(E_z*sat(ePred,psiZ)) @ linalg.pinv(np.diag(ePred)) #CMSVSF upper gain
        
        
            if modelType=="CV":
                if psiY_opt[0,0] <psiY[0] and psiY_opt[1,1] < psiY[1]:
                    Kl = P_Pred_21@H_1.T@S_inv #KF lower gain
                else:
                    Kl = np.diag(E_y * sat(M @ePred,psiY)) @ linalg.pinv(np.diag(M@ePred)) @ M #CMSVSF lower gain
            elif modelType=="CT":
                if psiY_opt[0,0] <psiY[0] and psiY_opt[1,1] < psiY[1] and psiY_opt[2,2] < psiY[2]:
                    Kl = P_Pred_21@H_1.T@S_inv #KF lower gain
                else:
                    Kl = np.diag(E_y * sat(M @ePred,psiY)) @ linalg.pinv(np.diag(M@ePred)) @ M #CMSVSF lower gain
            elif modelType == "CA":
                if psiY_opt[0,0] <psiY[0] and psiY_opt[1,1] < psiY[1] and psiY_opt[2,2] < psiY[2] and psiY_opt[3,3] < psiY[3]:
                    Kl = P_Pred_21@H_1.T@S_inv #KF lower gain
                else:
                    Kl = np.diag(E_y * sat(M @ePred,psiY)) @ linalg.pinv(np.diag(M@ePred)) @ M #CMSVSF lower gain
            #K = np.block([Ku,Kl]).T #stacked gain
            K = np.vstack((Ku,Kl))      
            P_c = P_Pred - K@H@P_Pred - P_Pred@H.T@K.T + K@S@K.T #update covariance

        else: #if using S.A. Gadsden's optimal SVSF
            if modelType=="CV":
                if psiZ_opt[0,0] < psiZ[0] and psiZ_opt[1,1] < psiZ[1] and psiZ_opt[2,2] < psiZ[2] and psiZ_opt[3,3]<psiZ[3]:
                    K = P_Pred_11@H_1.T@S_inv#KF upper gain
                else:
                    K = linalg.inv(H_1)@ np.diag(E_z*sat(ePred,psiZ)) @ linalg.pinv(np.diag(ePred)) #CMSVSF upper gain
            
            elif modelType=="CT":
                if psiZ_opt[0,0] < psiZ[0] and psiZ_opt[1,1] < psiZ[1] and psiZ_opt[2,2] < psiZ[2] and psiZ_opt[3,3]<psiZ[3] and psiZ_opt[4,4]<psiZ[4]:
                    K = P_Pred_11@H_1.T@S_inv #KF upper gain
                else:
                    K = linalg.inv(H_1)@ np.diag(E_z*sat(ePred,psiZ)) @ linalg.pinv(np.diag(ePred)) #CMSVSF upper gain
            
            P_c = (np.eye(n) - K@H)@P_Pred@(np.eye(n)-K@H).T + K@R@K.T #update covariance
            
               
        xPosBL = psiZ_opt[0,0]
        yPosBL = psiZ_opt[1,1]
        xVelBL = psiY_opt[0,0]
        yVelBL = psiY_opt[1,1]
        xAccBL = psiY_opt[2,2]
        yAccBL = psiY_opt[3,3]
        omegaBL = psiY_opt[4,4]
        
        BL_Vec = np.array([xPosBL,yPosBL,xVelBL,yVelBL,xAccBL,yAccBL,omegaBL])
        
        P_Tilda = K@(middleTerm - np.outer(ePred,ePred))@K.T #co-variance due to measurement origin uncertainty
        P_PostNew = Beta0*P_Pred + (1-Beta0)*P_c + P_Tilda #updated state co-variance
        P_PostNew = 0.5*(P_PostNew + P_PostNew.T) #for numerical stability
        
        xPostNew = xPred + K@ePred   #updated state estimate
        ePost = (np.eye(m)-H@K)@ePred #update a-posteriori measurement error
        
        #store updated state estimate, updated co-variance, and updated probability of track existence
        self.xPost = xPostNew
        self.P_Post = P_PostNew
        self.ePost = ePost
        self.pCurrent = pCurrent
        self.BL_Vec = BL_Vec
        
        return xPostNew,P_PostNew
        
    
    def IPDASVSF(self,measSet,MP,PD,PG,lambdaVal,sensorPos, T_mat, gammaZ, gammaY, psiZ, psiY):
        #performs data association, filtering, and updates probability of track existence
        
        #Inputs: measSet = array of gated measurements
        #       lambdaVal = parameter for clutter distribution
        #       PG = gate probability
        #       PD = probability of detection
        #       sensorPos = sensor position if using radar
        
        #outputs: xPostNew = updated state
        #         P_PostNew = updated co-variance
        #         pCurrent = updated probability of track existence
        
        xPost = self.xPost #latest state estimate
        P_Post = self.P_Post #latest co-variance
        ePost = self.ePost #latest a-posteriori measurement error
        pCurrent = self.pCurrent #latest track existence probability
        modelType = self.modelType #filter model
        sensor = self.sensor #type of sensor
        Ts = self.Ts #sampling time
        Q = self.Q #process noise co-variance
        R = self.R #measurement noise co-variance
        G = self.G #input gain matrix
        H = self.H #measurement matrix
        gatedArr = self.gateArr #binary array, if element i is 1, that means measurement i lies in the gate of this track 
        
        n = xPost.shape[0] #dimension of state vector
        m = measSet.shape[0] #dimension of measurement matrix
        m_k = int(sum(gatedArr)) #number of gated measurements
        gamma = chi2.ppf(PG, df=m) #gating threshold from PG
        
        #Predict step
        if modelType=="CV":
            xPred,F_k = CVModel(xPost,np.diag(np.array([0,0,0])),Ts) #predict using CV model
        elif modelType == "CA":
            xPred,F_k = CAModel(xPost,np.diag(np.array([0,0,0])),Ts) #predict using CV model
        elif modelType=="CT":
            xPred,F_k,LinF = CTModel(xPost, np.diag(np.array([0,0,0])), Ts) #predict using CT model
            F_k = LinF
        
        #predict co-variance
        if Q.shape[0]>3:
            P_Pred = F_k@P_Post@F_k.T + Q
        else:
            P_Pred = F_k@P_Post@F_k.T + G@Q@G.T
        
        P_Pred = 0.5*(P_Pred + P_Pred.T) #for numerical stability 
        
        pVec = np.array([pCurrent,1-pCurrent]) 
        pPredVec = MP.T@pVec#predict track existence probability
        pPred = pPredVec[0] #probability of track existence, other element in pPred is the probability of no track existence
        
        if sensor=="Radar": 
            rangePred = getRange(xPred,sensorPos) #predict measurement using non-linear model
            anglePred = getAngle(xPred,sensorPos)
            
            m=2
            zPred = np.zeros((m,))
            zPred[0] = rangePred #store into a vector
            zPred[1] = anglePred
            
            H = obtain_Jacobian_H(xPred, sensorPos) #obtain Jacobian matrix
        else: #if LIDAR
            zPred = H@xPred #predict measurement
            
        S = H@P_Pred@H.T + R #innovation co-variance
        S_inv = np.linalg.inv(S) #innovation co-variance inverse
                
        cn = math.pi #term used to get the volume of the gated region ellipsoid
        V_k = cn*math.sqrt(abs(np.linalg.det(gamma*S))) #volume of gated region
        
        if lambdaVal != 0: #if lambda is non-zero, compute mHat as follows
            mHat = lambdaVal*V_k
        else:
            mHat = 1- PD*PG*pPred #otherwise calculate mHat using this formula
        
        xMeas = measSet[0,:] #x component of measurements in gate
        yMeas = measSet[1,:] #y component of measurements in gate
        gateMeas = np.vstack((xMeas[gatedArr==1],yMeas[gatedArr==1])) #measurements in the gate stacked
        
        abs_det = abs(np.linalg.det(2*math.pi*S)) #term used to get the likelihood ratio 
        #T =   1/(  ((2*math.pi)**(m/2)) *math.sqrt(abs_det_S))
        T = 1/(math.sqrt(abs_det)) #term used to get the likelihood ratio
        innovations = np.zeros((m,m_k)) #innovation vectors
        
        prob_z_i_Vals = np.zeros((m_k,)) #likelihood function of measurements
        for i in range(m_k):
            z_i = gateMeas[:,i] #ith gated measurement
            nu_i = z_i-zPred #innovation
            innovations[:,i] = nu_i
            
            G_i = T*math.exp(-.5*nu_i.T@S_inv@nu_i) #likelihood function of measurement
            
            prob_z_i = G_i/PG #probability of measurement
            prob_z_i_Vals[i]=prob_z_i
          #  probZ_Sum = probZ_Sum+ prob_z_i
        probZ_Sum = sum(prob_z_i_Vals) #sum of probabilities 
        
        if m_k==0:
            delta_k = PD*PG #if m_k is zero use this formula to get the delta_k term
        else:
            delta_k = PD*PG- PD*PG*(V_k/mHat)*probZ_Sum #otherwise, apply this formula to obtain delta_k
            
        pCurrent = ((1-delta_k)/(1-delta_k*pPred))*pPred #update probability of track existence
        
        Beta0 = (1-PD*PG)/(1-delta_k) #probability of no measurement belonging to a target
        BetaVals = np.zeros((m_k,)) #association probabilities for measurements
        #compute association probabilities
        for i in range(m_k):
            prob_z_i = prob_z_i_Vals[i] #probability of z_i from earlier
            Beta_i = (PD*PG*(V_k/mHat)*prob_z_i)/(1-delta_k) #association probability of z_i belonging to the target
            BetaVals[i] = Beta_i
            
        nu_k = np.zeros((m,)) #weighted innovation
        middleTerm = np.zeros((m,m)) #term part of the co-variance update formula
        for i in range(m_k): #for each gated measurement
            nu_i=innovations[:,i] #ith innovation
            Beta_i = BetaVals[i] #association probability
            nu_k = nu_k + Beta_i*nu_i #weighted innovation
            
            middleTerm = middleTerm + Beta_i*np.outer(nu_i, nu_i) #term used to get a co-variance term
            
        H_1 = H[0:m,0:m] #sub-matrix of measurement matrix
        Psi_k = T_mat@F_k @linalg.inv(T_mat) #perform transformation
        
        ePred = nu_k #a-priori measurement error set to combined innovation
        ePred[ePred==0]=1E-1000
        
        #extract sub-matrices
        Psi_12 = Psi_k[0:m,m:n]
        Psi_22 = Psi_k[m:n,m:n]
        
        M = Psi_22 @ linalg.pinv(Psi_12)  
        
        E_z = abs(ePred) + gammaZ@abs(ePost)
        E_y = abs(M @ ePred) + gammaY@abs(linalg.pinv(Psi_12) @ePred)
        
        Ku = linalg.inv(H_1)@ np.diag(E_z*sat(ePred,psiZ)) @ linalg.pinv(np.diag(ePred)) #upper gain
        
        Kl = np.diag(E_y * sat(M @ePred,psiY)) @ linalg.pinv(np.diag(M@ePred)) @ M #CMSVSF lower gain
            
        K = np.vstack((Ku,Kl))  
        
        P_c = P_Pred - K@H@P_Pred - P_Pred@H.T@K.T + K@S@K.T #update covariance
        P_Tilda = K@(middleTerm - np.outer(ePred,ePred))@K.T #co-variance due to measurement origin uncertainty
        P_PostNew = Beta0*P_Pred + (1-Beta0)*P_c + P_Tilda #updated state co-variance
        P_PostNew = 0.5*(P_PostNew + P_PostNew.T) #for numerical stability
        
        xPostNew = xPred + K@ePred   #updated state estimate
        ePost = (np.eye(m)-H@K)@ePred #update a-posteriori measurement error
        
        #store updated state estimate, updated co-variance, and updated probability of track existence
        self.xPost = xPostNew
        self.P_Post = P_PostNew
        self.ePost = ePost
        self.pCurrent = pCurrent
        
        return xPostNew,P_PostNew
    
    def stackState(self,k): 
        #stores the state estimate from time step k
        xEsts = self.xEsts #array of state estimates
        xEsts[:,k] = self.xPost #include state estimate in the array
        
    def stackBL(self,k):
        BLs = self.BLs
        BLs[:,k] = self.BL_Vec
        
        self.BLs = BLs
        
       
    
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
        F_k = np.array([[1,0,math.sin(omega_k*Ts)/omega_k,(math.cos(omega_k*Ts)-1)/omega_k,0,0,0],
                       [0,1,(1-math.cos(omega_k*Ts))/omega_k,math.sin(omega_k*Ts)/omega_k,0,0,0],
                       [0,0, math.cos(omega_k*Ts), -math.sin(omega_k*Ts),0,0,0],
                       [0,0, math.sin(omega_k*Ts), math.cos(omega_k*Ts),0,0,0],
                       [0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,1]])
        
        LinF[0,0] = 1
        LinF[0,2] = math.sin(Ts*omega_k)/omega_k
        LinF[0,3] = (math.cos(Ts*omega_k)-1)/omega_k
        LinF[0,6] = (Ts*vx_k*math.cos(Ts*omega_k))/omega_k - (vy_k*(math.cos(Ts*omega_k) - 1))/omega_k**2 - (vx_k*math.sin(Ts*omega_k))/omega_k**2 - (Ts*vy_k*math.sin(Ts*omega_k))/omega_k;
        
        LinF[1,1] = 1
        LinF[1,2] =  -(math.cos(Ts*omega_k) - 1)/omega_k
        LinF[1,3] = math.sin(Ts*omega_k)/omega_k
        LinF[1,6]  =  (vx_k*(math.cos(Ts*omega_k) - 1))/omega_k**2 - (vy_k*math.sin(Ts*omega_k))/omega_k**2 + (Ts*vy_k*math.cos(Ts*omega_k))/omega_k + (Ts*vx_k*math.sin(Ts*omega_k))/omega_k;
        
        LinF[2,2] = math.cos(Ts*omega_k)
        LinF[2,3] = -math.sin(Ts*omega_k)
        LinF[2,6] = - Ts*vy_k*math.cos(Ts*omega_k) - Ts*vx_k*math.sin(Ts*omega_k);
        
        LinF[3,2] = math.sin(Ts*omega_k)
        LinF[3,3] = math.cos(Ts*omega_k)
        LinF[3,6] = Ts*vx_k*math.cos(Ts*omega_k) - Ts*vy_k*math.sin(Ts*omega_k);
       
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
        
   


def getRange(xVec,sensorPos): #obtains the range using the radar measurment model
    #Inputs: xVec = state vector
    #       sensorPos = position of radar
    
    #Outputs: R = range

    xS = sensorPos[0] #x-position of radar
    yS = sensorPos[1] #y-position of radar
    
    x_k = xVec[0] #x-position of target
    y_k = xVec[1] #y-position of target
    
    R = math.sqrt((x_k-xS)**2 + (y_k-yS)**2) #range formula
    
    return R

def getAngle(xVec,sensorPos): #obtains the angle from the radar measurement model
    #Inputs: xVec = state vector
    #       sensorPos = position of radar
    
    #Outputs: theta = angle
    
    xS = sensorPos[0]#x-position of radar
    yS = sensorPos[1]#y-position of radar
    
    x_k = xVec[0]#x-position of target
    y_k = xVec[1]#y-position of target
    
    theta = math.atan2(y_k-yS, x_k-xS) #angle formula 
    
    return theta

def obtain_Jacobian_H(xVec,sensorPos): #obtains the Jacobian of the radar measurement model
    #Inputs: xVec = state vector
    #       sensorPos = position of radar
    
    #Outputs: H = Jacobian of measurement model

    n = xVec.shape[0] #number of states
    
    xS = sensorPos[0] #X and Y coordinates of sensor
    yS = sensorPos[1]
    
    x_k = xVec[0] #X and Y position of target
    y_k = xVec[1]
    
    m = 2 #number of measurements
    
    H = np.zeros((m,n)) #allocate Jacobian matrix
    
    #obtain Jacobian matrix entries
    denomTerm = math.sqrt((x_k-xS)**2 + (y_k-yS)**2)
    denomTermSqr = denomTerm**2

    #compute entries of Jacobian
    H[0,0] = (x_k-xS)/denomTerm
    H[0,1] = (y_k-yS)/denomTerm
    H[1,0] = -(y_k-yS)/denomTermSqr
    H[1,1] = (x_k-xS)/denomTermSqr
    
    return H

def sat(vec,psi):
    outputVec = vec/psi #element-wise division
    absOutputVec = abs(outputVec)
    overOne = absOutputVec>1 #obtain elements larger than 1 in absolute value
    outputVec[overOne] = np.sign(outputVec[overOne]) #apply the sign function to the elements greater than 1
    
    return outputVec

