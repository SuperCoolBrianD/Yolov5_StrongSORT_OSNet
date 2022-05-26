# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:29:03 2022

@author: salma
"""
import numpy as np
from math import*
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.linalg import sqrtm
from scipy.stats.distributions import chi2
import random
# from track import*
from track_MM import*

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
    


def getRange(xVec,sensorPos):
    xS = sensorPos[0]
    yS = sensorPos[1]
    
    x_k = xVec[0]
    y_k = xVec[1]
    
    R = sqrt((x_k-xS)**2 + (y_k-yS)**2)
    
    return R

def getAngle(xVec,sensorPos):
    xS = sensorPos[0]
    yS = sensorPos[1]
    
    x_k = xVec[0]
    y_k = xVec[1]
    
    theta = atan2(y_k-yS, x_k-xS)
    
    return theta

def obtain_Jacobian_H(xVec,sensorPos):
    n = xVec.shape[0] #number of states
    
    xS = sensorPos[0] #X and Y coordinates of sensor
    yS = sensorPos[1]
    
    x_k = xVec[0] #X and Y position of target
    y_k = xVec[1]
    
    m = 2 #number of measurements
    
    H = np.zeros((m,n)) #allocate Jacobian matrix
    
    #obtain Jacobian matrix entries
    denomTerm = sqrt((x_k-xS)**2 + (y_k-yS)**2)
    denomTermSqr = denomTerm**2

    H[0,0] = (x_k-xS)/denomTerm
    H[0,1] = (y_k-yS)/denomTerm
    H[1,0] = -(y_k-yS)/denomTermSqr
    H[1,1] = (x_k-xS)/denomTermSqr
    
    return H

def generateTrajectory(x0,modelList, xVelList, yVelList, xAccList, yAccList, omegaList,timeList,covList,startSampleInit,Ts,N):
    #Generate the true trajectory of a target
    
    #Inputs: x0 = initial state
    #        modelList = list of models, each element specifies a model used to generate the trajectory of a segement
    #        xVelList = list of specified x velocities for each segment
    #        yVelList = list of specified y velocities for each segment
    #        omegaList = list of specified turn-rates for each segment
    #        timeList = specifies the end time of each segment
    #        covList = process noise co-variance of segment i
    
    #Outputs: trueTraj = arrays of true states
    #         outputVec last state of segment 1
    #         t_0 = end time of segment 1
    
    
    n = int(x0.shape[0])
    #Q1 = np.array([[sigma_v**2,0],[0,sigma_v**2]])
    #Q2 = np.array([[sigma_v**2,0,0],[0,sigma_v**2,0],[0,0,sigma_v**2]])
    
    if hasattr(timeList, "__len__") == False: #if there is one segment
    #    N = int(timeList/Ts)
        numSeg = 1
    else:                                      #if multiple segments
     #   N = int(round(timeList[-1]/Ts))
        numSeg = len(timeList) #number of segments
        
    trueTraj = np.zeros((n,N+1)) #allocate a array to store the true states
    xk = x0 #set x_k to the initial state x0
    
    trueTraj[:,startSampleInit] = xk #store initial state
    
    for i in range(numSeg): #for each segment
        if hasattr(timeList, "__len__") == False: #if 1 segment
            t_i = timeList
            endTime = t_i
            Q = covList
            omega = omegaList
            modelType = modelList
            vx = xVelList
            vy = yVelList
            ax = xAccList
            ay = yAccList
        else:                                      #if multiple segments
            t_i = timeList[i] #end time of segment i
            endTime = t_i
            Q = covList[i] #process noise covariance of segment i
            omega=omegaList[i] #turn-rate of segment i
            modelType = modelList[i] #model for segment i
            vx = xVelList[i] #specified x velocity for segment i
            vy = yVelList[i] #specified y velocity for segment i
            ax = xAccList[i]
            ay = yAccList[i]
            
        if vx!=0:
            xk[2] = vx
        if vy!=0:
            xk[3] = vy
        if ax!=0:
            xk[4] = ax
        if ay!=0:
            xk[5] = ay
        
        xk[6] = omega #store the specified turn-rate into the state vector
        numSamples = round(endTime/Ts) #number of samples in segment i
        
        if i==0:
            startSample =startSampleInit+1 #start sample for segment if its the first
        else:
            startSample = int(timeList[i-1]/Ts)+1 #state sample if its the second or above
                
        
        for k in range(startSample, numSamples+1): #for each k from startSample to numSamples, obtain the ground truth states
            if modelType == "CV": #if the model is CV
                xkNew = CVModel(xk,Q,Ts)[0] #obtain next state using the CV model
                trueTraj[:,k] = xkNew #store the state
                xk = xkNew
            elif modelType == "CA":
                xkNew = CAModel(xk,Q,Ts)[0]
                trueTraj[:,k] = xkNew #store the state
                xk = xkNew
            elif modelType == "CT":
                xkNew = CTModel(xk,Q,Ts)[0]
                trueTraj[:,k] = xkNew #store the state
                xk = xkNew
        if i==0:
            outputVec = xk
            t_0 =endTime
                
    return trueTraj, outputVec, t_0

def generateClutter(xLims,yLims,lambdaVal):
    #generates clutter points in a specific region specified by xLims and yLims
    
    #Inputs: xLims = limits for the x range in the coverage region
    #        yLims = limits for the y range in the coverage region
    #        lambdaVal= parameter for generating the number of clutter points
    
    xMin = xLims[0] #min. x value in the region
    xMax = xLims[1] #max. x value in the region
    yMin = yLims[0] #min. y value in the region
    yMax = yLims[1] #max. y value in the region

    Vol = (xMax-xMin)*(yMax-yMin) #volume of the region
    
    numPoints = np.random.poisson(Vol*lambdaVal) #number of points generated from a Poisson distributed
    
    xPoints = np.random.uniform(xMin,xMax+1,numPoints) #uniformly generate the random x points
    yPoints = np.random.uniform(yMin,yMax+1,numPoints) #uniformly generate the random y points
    
    clutterMeas = np.vstack((xPoints,yPoints)) #stack the points to create a array of 2D points
    
    return clutterMeas

def gating(trackList,lastTrackIdx,PG, MP, maxVals,sensorPos,measSet):
    #takes the measurements obtained from the current time step or frame, then selects them based on the stochastic distance
    
    #numTracks = len(trackList)
    numTracks = lastTrackIdx +1 #total number of tracks to this point
    numMeas = measSet.shape[1] #number of measurements
    
    gateMat = np.zeros((numTracks,numMeas)) #binary matrix, the index of the row specifies a track, the column specifies the measurement
    for i in range(numTracks): #for each track
        track_i = trackList[i]
        status = track_i.status
        isMM = track_i.isMM
         
        if status == 1 or status == 2: #if the track is tenative or confirmed
            if isMM==False:
                #obtain the track's state, co-variance, the model for the filter, sampling time, process noise co-variance, measurement noise co-variance, sensor type, input gain matrix, and measurement matrix
                xPost_i = track_i.xPost
                P_Post_i = track_i.P_Post
                modelType_i = track_i.modelType
                Ts = track_i.Ts
                Q_i = track_i.Q
                R = track_i.R
                sensor =track_i.sensor
                G_i= track_i.G
                H = track_i.H
                gateArr = gatingSingle(xPost_i,P_Post_i,G_i,H,Q_i,R,PG,Ts,modelType_i,sensorPos,sensor,measSet)
            else:
                #if a IMM algorithm is being used
                '''
                xPosts = track_i.xPosts
                P_Posts = track_i.P_Posts
                G_List = track_i.G_List
                Q_List = track_i.Q_List
                H = track_i.H
                R = track_i.R
                models = track_i.models
                sensor =track_i.sensor
                '''
                
                track_i.IMMIPDA_Initialize(MP,maxVals) #initialize IMM estimator of track
                modeTrackList_i = track_i.modeTrackList #mode conditioned tracks of track i
                gateArr, modeTrackList_i = gatingMM(modeTrackList_i,PG,sensorPos,measSet) #perform gating
                track_i.modeTrackList = modeTrackList_i #store updated info for the mode-conditioned tracks
            
            track_i.gateArr = gateArr #store gateArr as a property for track_i
            gateMat[i,:] = gateArr #store gateArr into the matrix
            trackList[i] = track_i #store the updated track into the trackList
        
    sumGateMat = np.sum(gateMat,axis=0) #take the sum of the matrix to see which measurments were not gated
    nonGatedIdx = np.nonzero(sumGateMat==0) #indices of measurements not gated
    
    unassignedMeas = measSet[:,nonGatedIdx] #obtain the measurements not gated, also known as unassigned measurements
    unassignedMeas = np.vstack((measSet[:,nonGatedIdx][0],measSet[:,nonGatedIdx][1]))
    
    return trackList, unassignedMeas

def gatingMM(modeTrackList,PG,sensorPos,measSet):
    #Performs gating for IMM algorithms by taking the union of validated measurements from each model
    r = len(modeTrackList) #number of models

    nm = measSet.shape[1] #number of measurements
    gatedMat = np.zeros((r,nm)) #each row is a binary array, where 1 indicates that a measurement is gated
    
    for i in range(r): #apply gating using each model
        x0_i = modeTrackList[i].xPostNew #mode-conditioned initial state for mode j
        P0_i = modeTrackList[i].P_PostNew#mode-conditioned initial state co-variance for mode j
        G_i = modeTrackList[i].G #input gain for mode j
        Q_i = modeTrackList[i].Q #Q matrix for mode j
        H = modeTrackList[i].H
        R = modeTrackList[i].R
        Ts = modeTrackList[i].Ts
        sensor = modeTrackList[i].sensor
        modelType = modeTrackList[i].modelType #motion model for mode j

        gatedArr=gatingSingle(x0_i,P0_i,G_i,H,Q_i,R,PG,Ts,modelType,sensorPos,sensor,measSet)
        gatedMat[i,:] = gatedArr
        
    sumMat = np.sum(gatedMat,axis=0)
    
    sumMat[sumMat>=1] = 1 #take the union for gating
    gatedArr = sumMat
    
    for i in range(r):
        modeTrackList[i].gateArr = gatedArr #store the union to each mode-matched track in order for the IMM algorithm to use it
    
    return gatedArr, modeTrackList
    

def gatingSingle(xPost,P_Post,G,H,Q,R,PG,Ts,modelType,sensorPos,sensor,measSet):
    nm = measSet.shape[1] #number of measurements
    m = H.shape[0] #number of measured states
    gamma = chi2.ppf(PG, df=m) #threshold for gating based on the gate probability PG
    
    #Predict the state according to the model
    if modelType=="CV":
        xPred,F_k = CVModel(xPost,np.diag(np.array([0,0,0])),Ts)
    elif modelType=="CA":
        xPred,F_k = CAModel(xPost,np.diag(np.array([0,0,0])),Ts)
    elif modelType=="CT":
        xPred,F_k,LinF = CTModel(xPost, np.diag(np.array([0,0,0])), Ts)    
        F_k = LinF
    
    #predict the co-variance
    if Q.shape[0]>3:
        P_Pred = F_k@P_Post@F_k.T + Q
    else:
        P_Pred = F_k@P_Post@F_k.T + G@Q@G.T
    
    P_Pred = 0.5*(P_Pred + P_Pred.T) #for numerical stability

    #predict the measurement based on sensor
    if sensor=="Radar":
        rangePred = getRange(xPred,sensorPos) #predict measurement using non-linear model
        anglePred = getAngle(xPred,sensorPos)
        
        m=2
        zPred = np.zeros((m,))
        zPred[0] = rangePred #store into a vector
        zPred[1] = anglePred
        
        H = obtain_Jacobian_H(xPred, sensorPos) #obtain Jacobian matrix
    else: #if LIDAR
        zPred = H@xPred 

    S = H@P_Pred@H.T + R #innovation co-variance
    S_inv = np.linalg.pinv(S)
    
    gateArr = np.zeros((nm,)) #binary array, if measurement j falls in the gate, set it to 1
    
    for j in range(nm): #for each measurement j
        z_j = measSet[:,j] #get measurement
        nu_j = z_j-zPred
        dist = nu_j.T @ S_inv @nu_j #compute stochastic distance
        
        if dist<=gamma: #if below the threshold, set element j of gateArr to 1
            gateArr[j] = 1
    
    return gateArr
 

def initiateTracks(trackList,lastTrackIdx,measSet, maxVel,maxAcc,maxOmega,G,H,Q,R,modelType,Ts,pInit, startSample,sensor,N):
    #initiates a track for each measurement in measSet
    
    isMM = False
    numMeas = measSet.shape[1]
    #trackList = []
    if numMeas!=0: #if the number of measurements in nonzero, initiate tracks 
        j = lastTrackIdx #index of latest track
        for i in range(numMeas):
            j=j+1 #index of new track
            z_i = measSet[:,i] #ith measurement 
            
            track_j = track(z_i,G,H,Q,R,maxVel, maxAcc,maxOmega,pInit,startSample,Ts,modelType,sensor,isMM,N) #initiate jth track 
            trackList[j] = track_j #store track at position j in trackList
           
        lastTrackIdx = j #index of last track
    return trackList,lastTrackIdx

def initiateTracksMM(trackList,lastTrackIdx,measSet, maxVals,G_List,H,Q_List,R,models,filters,Ts,pInit, startSample,sensor,N):
    #initiates a track for each measurement in measSet
    
    numMeas = measSet.shape[1]
    
    #trackList = []
    if numMeas!=0: #if the number of measurements in nonzero, initiate tracks 
        j = lastTrackIdx #index of latest track
        for i in range(numMeas):
            j=j+1 #index of new track
            z_i = measSet[:,i] #ith measurement 
            
            trackMM_j = track_MM(z_i,G_List,H,Q_List,R,maxVals,pInit,startSample,Ts,models,filters,sensor,N) #initiate jth track 
            trackList[j] = trackMM_j #store track at position j in trackList
           
        lastTrackIdx = j #index of last track
    return trackList,lastTrackIdx


def updateStateTracks(trackList,lastTrackIdx, filterType,measSet, maxVals,lambdaVal,MP,PG,PD,sensorPos,T_mat, gammaZ, gammaY, psiZ, psiY,k):
    #updates the state and co-variance of each track using a filter
    
    #numTracks = len(trackList)
    numTracks = lastTrackIdx + 1 
#    if numTracks>0:
    for i in range(numTracks): #for each track update the state
        track_i = trackList[i] 
        status = track_i.status
        
        if status == 1 or status == 2: #if the track is tenative or confirmed, update the state
            gateArr_i = trackList[i].gateArr #binary array that indicates which measurements are gated
            gateMeasIdx = np.nonzero(gateArr_i==1) #indices of gated measurements
            
            for j in range(numTracks): #for each other track
                if i!=j and (trackList[j].status==1 or trackList[j].status==2): #for every other track, remove associated measurements of track i that fall in another track's gate
                    gateArr_j = trackList[j].gateArr #gatedArr of track j
                    gateArr_j[gateMeasIdx] = 0 #remove gated measurements of track i that lie in the gate of another track
                    trackList[j].gateArr = gateArr_j #set that to the new gateArr
                    
                    '''
                    if trackList[j].isMM == True: #if the track uses a IMM filter
                        modeTrackList = trackList[j].modeTrackList #mode-matched tracks
                        r = len(modeTrackList) #number of models
                        
                        for kk in range(r):
                            track_k = modeTrackList[kk] #kth mode-matched track
                            gateArr_k = track_k.gateArr #gated binary array of mode-matched track
                            gateArr_k[gateMeasIdx]=0
                            track_k.gateArr = gateArr_k #replace it with new
                            modeTrackList[kk] = track_k
                        
                        trackList[j].modeTrackList = modeTrackList
                    '''
        
            #update the state using a filter
            if filterType == "PDAKF": 
                track_i.PDAKF(measSet,lambdaVal,PG,PD,sensorPos)
            elif filterType == "IPDAKF":
                track_i.IPDAKF(measSet,MP,PD,PG,lambdaVal,sensorPos)
            elif filterType == "IPDASVSF":
                track_i.IPDASVSF(measSet, MP, PD, PG, lambdaVal, sensorPos, T_mat, gammaZ, gammaY, psiZ, psiY)
            elif filterType == "IPDAGVBLSVSF":
                track_i.IPDAGVBLSVSF(measSet, MP, PD, PG, lambdaVal, sensorPos, T_mat, gammaZ, gammaY, psiZ, psiY)
                track_i.stackBL(k)
            elif filterType == "IMMIPDAKF":
                #track_i.IMMIPDAKF(measSet,MP,PD,PG,lambdaVal,maxVals,sensorPos)
                track_i.IMMIPDA_Filter(measSet,MP,PD,PG,lambdaVal,maxVals,sensorPos)
            track_i.stackState(k) #store the state estimate into the track's array
            
            trackList[i] = track_i #store updated track i
    return trackList

'''
def updateStateTracksMM(trackList,lastTrackIdx, filterType,measSet, maxVals,lambdaVal,MP,PG,PD,sensorPos,T_mat, gammaZ, gammaY, psiZ, psiY,k):
    #Performs gating, data association, and state estimation for each track
    
    numTracks = lastTrackIdx + 1 
    numMeas = measSet.shape[1] #number of measurements
    
    gateMat = np.zeros((numTracks,numMeas)) #binary matrix, the index of the row specifies a track, the column specifies the measurement
    for i in range(numTracks):
        track_i = trackList[i] 
        status = track_i.status
        
        if status == 1 or status == 2: #if the track is tenative or confirmed
            gateArr = track_i.IPDAGVBLSVSF(measSet, MP, PD, PG, lambdaVal, sensorPos, T_mat, gammaZ, gammaY, psiZ, psiY)
            track_i.stackBL(k)
            gateMat[i,:] = gateArr #store gateArr into the matrix
            trackList[i] = track_i #store the updated track into the trackList
            
    sumGateMat = np.sum(gateMat,axis=0) #take the sum of the matrix to see which measurments were not gated
    nonGatedIdx = np.nonzero(sumGateMat==0) #indices of measurements not gated
     
    unassignedMeas = measSet[:,nonGatedIdx] #obtain the measurements not gated, also known as unassigned measurements
    unassignedMeas = np.vstack((measSet[:,nonGatedIdx][0],measSet[:,nonGatedIdx][1]))
    
    return 1
'''
            
def updateTracksStatus(trackList,lastTrackIdx,delTenThr, delConfThr, confTenThr,k):
    #implements the track manager, which updates the status of each track
    
    numTracks= lastTrackIdx+1
    for i in range(numTracks): #for each track
        status = trackList[i].status #current status
        p = trackList[i].pCurrent #current probability of track existence
        
        #0 indicates the track is deleted, 1 means its tenative, and 2 indicates confirmed
        if status==1: #if the track is tenative
            if p<delTenThr: #if under the deletion threshold
                trackList[i].status = 0 #remove tenative track, set the status as deleted
                trackList[i].endSample = k
            elif p>=confTenThr: #if above the confirmation threshold
                trackList[i].status = 2 #confirm track, set that status as confirmed
        elif status==2: #if the track is confirmed
            if p<delConfThr: #if below the deletion threshold
                trackList[i].status = 0  #removed confirmed track
                trackList[i].endSample = k #store the last sample/frame processed
                
                
                
    return trackList
            

def computeRMSE(truth,est):
    #computes the scalar RMSE between 2 arrays
    error = truth - est#subtract arrays to get error
    sumErr = sum(error**2) 
    N = error.shape[0]
    
    RMSE = sqrt(sumErr/N) #apply RMSE formula

    return RMSE

def RMSEPlot(errPlotsX,errPlotsY):
    #obtains the RMSE vectors for the X and Y variable, which can be e.g. X and Y position for instance
    #errPlotsX is a array which contains the error of a x state, each row contains the error in a specific Monte Carlo run
    #errPlotsY is a array which contains the error of a y state, each row contains the error in a specific Monte Carlo run
    
    '''
    #errX_Squared = errX**2
    #errY_Squared = errY**2
    
    #errMat = errX_Squared + errY_Squared
    
    #sumErrVec = errMat.sum(axis=0)
    
    #rmsePlot = sumErrVec**.5
    '''
    
    N = errPlotsX.shape[1] #number of samples
    numRuns = errPlotsX.shape[0] #number of Monte Carlo runs
    rmsePlot = np.zeros((N,1))
    xErrSqr_Total = 0
    yErrSqr_Total = 0
    
    for k in range(N):
        errX_k = errPlotsX[:,k] #errors at sample k for x
        errY_k = errPlotsY[:,k] #errors at sample k for y
        
        sumSquaredErr_X_k = sum(errX_k**2)
        sumSquaredErr_Y_k = sum(errY_k**2)
        
        RMSE_k = sqrt((sumSquaredErr_X_k +sumSquaredErr_Y_k)/numRuns ) #RMSE at sample k
        rmsePlot[k] = RMSE_k #store it in the rmsePlot array
        
        xErrSqr_Total = xErrSqr_Total + sumSquaredErr_X_k
        yErrSqr_Total = yErrSqr_Total + sumSquaredErr_Y_k
        
    #obtain the scalar RMSEs
    scalarRMSE_X = sqrt(xErrSqr_Total/(numRuns*N))
    scalarRMSE_Y = sqrt(yErrSqr_Total/(numRuns*N))
    scalarRMSE_Total = sqrt((xErrSqr_Total+yErrSqr_Total)/(numRuns*N))
    
    return rmsePlot,scalarRMSE_X ,scalarRMSE_Y,scalarRMSE_Total 
