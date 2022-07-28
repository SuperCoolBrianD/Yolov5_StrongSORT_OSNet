# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:27:15 2022

@author: salma
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.linalg import sqrtm
from MTT_Functions import*
from track import*

plt.close('all')

from datetime import datetime
startTime = datetime.now()

filterType="IPDAKF" #tracking algorithm
useSVSF_G = False #use S.A. Gadsden's SVSF
modelType = "CV" #model for tracker, only single model trackers in this program
useNewTraj = 1 # 1 means to generate a new synthetic trajectory, 0 means to use a existing one from a file
isIMM = False

totalTime = 200 #total duration of simulation
numTargets = 2#number of targets

#Parameters
Ts = 1 #sampling time
sigma_w = 10 #measurement noise standard deviation in position
sigmaVel = (1/(Ts**2))*math.sqrt(sigma_w**2  + sigma_w**2) #velocity measurement standard deviation
sigmaOmega = .05
sigma_v = 1E-1 #process noise standard deviation
L1 = .16
L2 = .06
sigma_v_filt = sigma_v #process noise standard deviation for filter
maxVel = 10 #for initializing velocity variance for 1-point initialization
maxAcc = 0
omegaMax = math.radians(0) #for initializing turn-rate variance for 1-point initialization
maxVals = [maxVel,maxAcc,omegaMax]
numRuns =100#number of Monte Carlo Runs
N = int(totalTime/Ts) #number of samples in the timulation

t = np.arange(0,totalTime,Ts) #time vector for simulation for plotting

#Parameters for tracking in clutter
lambdaVal = 1E-4#parameter for clutter density
PD = .8#probability of target detection in a time step
#PG = .9999999999999999
PG = .9979 #gate probability
regionWidth = 100 #width of the clutter region surrounding a target
regionLength = 100 #length of the clutter region surrounding a target
thetaLims = [-math.pi,math.pi]
rangeLims = [0,10000]
pInit = .2 #initial probability of track existence

P_ii = .999 #for Markov matrix of IPDA
MP = np.array([[P_ii,1-P_ii],[1-P_ii,P_ii]]) #Markov matrix for IPDA

#Parameters for track management

useLogic=True
delTenThr = .05; #threshold for deleting a tenative track
confTenThr = .9; #threshold for confirming a tenative track
delConfThr = 0.005; #threshold for deleting a confirmed track

#For M/N logic based track management
N1 = 3
M2 = 3
N2 = 6
N3 = 5

errThr = 4000#threshold for error cases in which the track is lost
errThr2 = 1200
numErrCases = 0 #counts the number of error cases

#Parameter for prediction
predTraj = False
tP = 7

sensorPos = np.array([1000,500]) #position of radar sensor 
sensor = "Radar" #sensor type
rangeStd = 10 #standard deviation of range for radar
angleStd = .01 #standard deviation of angle for radar

Q_CV = np.diag([sigma_v**2,sigma_v**2,0]) #process noise co-variance for CV model
Q_CA = np.diag([sigma_v**2,sigma_v**2,0]) #process noise co-variance for CV model
Q_CT = np.diag([sigma_v**2,sigma_v**2, sigma_v**2]) #process noise co-variance for CT model

#For generating the trajectory of the 1st remote car, 7 variables below are required to obtain the ground truth trajectory
#x0_Target1 = np.array([0,0,27,27,0.5,0.5, math.radians(0)]) #initial state for the 1st target
x0_Target1 = np.array([6000,5000,-10,-10,0,0,0])
covListTarget1 = Q_CV #each entry is the process noise co-variance for each segment
xVelListTarget1 = 0 #if the velocity has to change to a specific value in a segment, change that
yVelListTarget1 = 0
xAccListTarget1 = 0 #if the velocity has to change to a specific value in a segment, change that
yAccListTarget1 = 0
omegaListTarget1 = math.radians(0) #turn-rate for each segment
modelListTarget1 = 'CV' # model for each segment
timesTarget1 =30 #time at which each segment concludes

#t1_start = 5 #time in which target 1 appears
t1_start = 0

if isinstance(covListTarget1, list)==False:
    t1_end = timesTarget1
else:
    t1_end = timesTarget1[-1] #time in which target 1 disappears
    
t1_startSample = int(t1_start/Ts) #sample number in which target 1 appears
t1_endSample = int(t1_end/Ts) #sample number in which target 2 disappears
N_t1 = t1_endSample-t1_startSample+1 #number of samples for the period in which the target is present

#For 2nd remote car
#x0_Target2 = np.array([-100,-100,-25,-25,-0.5,-0.5, math.radians(0)])
x0_Target2 = np.array([2000,2000,-10,-10,0,0, math.radians(0)])
covListTarget2 = Q_CV
xVelListTarget2 = 0
yVelListTarget2 = 0
xAccListTarget2 = 0 #if the velocity has to change to a specific value in a segment, change that
yAccListTarget2 = 0
omegaListTarget2 =math.radians(0)
modelListTarget2 = 'CV'
timesTarget2 = 40

t2_start = 15

if isinstance(covListTarget2, list)==False:
    t2_end = timesTarget2
else:
    t2_end = timesTarget2[-1] #time in which target 1 disappears

t2_startSample = int(t2_start/Ts)
t2_endSample = int(t2_end/Ts)
N_t2 =t2_endSample  -t2_startSample+1

#Filter parameters
if modelType=="CV":
    Q = np.diag(np.array([sigma_v_filt**2,sigma_v_filt**2,0**2]))
    G = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0],[0,0,0],[0,0,0],[0,0,0]]) 
    ''''
    Q_CV = L1*np.array([[(Ts**3)/3, 0, (Ts**2)/2, 0, 0],
                     [0,(Ts**3)/3, 0, (Ts**2)/2, 0 ],
                     [(Ts**2)/2, 0, Ts, 0, 0],
                     [0, (Ts**2)/2, 0, Ts, 0],
                    [0, 0, 0, 0, 0]])
    #Q=Q_CV
    '''
elif modelType=="CA":
    Q = np.diag(np.array([sigma_v_filt**2,sigma_v_filt**2,0**2]))
    G = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0], [1,0,0],[0,1,0],[0,0,0]]) 
    
elif modelType=="CT":
    Q = np.diag(np.array([sigma_v_filt**2,sigma_v_filt**2,sigma_v_filt**2]))
    G = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0],[0,0,0],[0,0,0],[0,0,Ts]])
    '''
    Q_CT = L1*np.array([[(Ts**3)/3, 0, (Ts**2)/2, 0, 0],
                     [0,(Ts**3)/3, 0, (Ts**2)/2, 0 ],
                     [(Ts**2)/2, 0, Ts, 0, 0],
                     [0, (Ts**2)/2, 0, Ts, 0],
                    [0, 0, 0, 0, (L2/L1)*Ts]])
    #Q=Q_CT
    '''
    
if useSVSF_G==True: #use S.A. Gadsden's SVSF
    #requires the number of states to equal the number of measurements
    R = np.diag(np.array([sigma_w**2,sigma_w**2, sigmaVel**2, sigmaVel**2, sigmaOmega**2])) #measurement co-variance
    H = np.eye(n) #measurement matrix
else: #if the number of measurements is less than the number of states
    if sensor=="Radar":
        R = np.diag(np.array([rangeStd**2,angleStd**2])) #if radar, use this measurement co-variance
        H = 0
    else:
        R = np.diag(np.array([sigma_w**2,sigma_w**2])) #measurement co-variance
        H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0]]) #measurement matrix
m = R.shape[0] #number of measurements

#UKF parameter
kappa = 1E-1000

#SVSF Parameters
n = x0_Target1.shape[0]
T = np.eye(n)
if useSVSF_G==True: #parameters for S.A.Gadsden's SVSF
    gammaZ = 0.0*np.eye(n)
    gammaY = gammaZ
    
    psi1 = 1.0
    psi2 = 8.0
    psi3 = 269.0
    psiZ = np.array([psi1,psi1,psi2,psi2,psi3])
    psiY = np.array([170.0,170.0,561.0])
else:
    #Parameters for the m<n SVSF, where m=# of measurements and n=# of states
    
    #gammaZ=0.0039*np.eye(2)
    #gammaY=0.0039*np.eye(3)
    #psiZ = np.array([1769.0,1769.0])
    #psiY = np.array([993.0,993.0,435.0])
    
    
    ##Best for CT
    #gammaZ=0.1*np.eye(m)
   # gammaY=0.1*np.eye(n-m)
   # psi1 = 1769.0
    #psi2 = 993.0
    #psi3 = 435.0
    #psi1 = 11
    #psi2 = 11
    #psi3 =300
    #gammaZ =0.56*np.eye(m)
    #gammaY = 0.56*np.eye(n-m)
    
    #psi1 = 10
    #psi2 = 10
    #psi3 = 1000
    #gammaZ =0.1*np.eye(m)
    #gammaY = 0.1*np.eye(n-m)
    """
    psi1 = 10
    psi2 = 100
    psi3 = 100
    gammaZ = .1*np.eye(m)
    gammaY = .1*np.eye(n-m)
    """
    
    #psi1= 657.0
    '''
    psi1 = 10
    psi2 = 20
    psi3 = 20
    gammaZ = 0.8*np.eye(m)
    gammaY = 0.8*np.eye(n-m)
    '''
    '''
    psi1 = math.inf
    psi2 = math.inf
    psi3 = math.inf
    gammaZ = .1*np.eye(m)
    gammaY = .1*np.eye(n-m)
    '''
    
    '''
    psi1 = 10*5
    psi2 = 75*2
    psi3 = 75
    gammaZ = 0.9*np.eye(m)
    gammaY = 0.9*np.eye(n-m)
    '''
    
    psi1 = 5
    psi2 = 50
    psi3 = 100
    psi4 = 100
    gammaZ = .9*np.eye(m)
    gammaY = .9*np.eye(n-m)
    
    ##Best for CV
    '''
    gammaZ = 0.778*np.eye(m)
    gammaY = 0.778*np.eye(n-m)
    psi1 = 9333.0
    psi2 = 40.0
    psi3 = 10 #arbitrary 
    '''
    
    psiZ = np.array([psi1,psi1])
    psiY = np.array([psi2,psi2,psi3,psi3,psi4])
    
SVSFParams = [psiZ,psiY,gammaZ,gammaY,T]
    
#generate true trajectories of targets
if useNewTraj==1:
    startSampleT1 = int(t1_start/Ts) 
    startSampleT2 = int(t2_start/Ts)
    
    xTargetCarTraj1 = generateTrajectory(x0_Target1, modelListTarget1, xVelListTarget1, yVelListTarget1, xAccListTarget1, yAccListTarget1, omegaListTarget1, timesTarget1, covListTarget1,startSampleT1, Ts,N)[0]
    xTargetCarTraj2 = generateTrajectory(x0_Target2, modelListTarget2, xVelListTarget2, yVelListTarget2, xAccListTarget2, yAccListTarget2, omegaListTarget2, timesTarget2, covListTarget2,startSampleT2, Ts,N)[0]
    
    """
    x0 = outputVec
    times = [timesEgo[1]-t_0,timesEgo[2]-t_0, timesEgo[3]-t_0]
    covList = [Q_CT,Q_CT,Q_CT]
    omegaList = [math.radians(4),math.radians(4),math.radians(4)]
    modelList = ["CT","CT","CT"]
    xVelList = [0,0,0]
    yVelList = [0,0,0]
    xPosEgo = xEgoCarTraj[0,:]
    yPosEgo = xEgoCarTraj[1,:]
    """
else:
    #xTrue =np.loadtxt('trueTraj3_Ts_1.txt', dtype=float)
    xEgoCarTraj = np.loadtxt('EgoCarTraj2.txt', dtype=float) #use saved trajectory from a .txt file
    
#ground truth x position, y position, x velocity, y velocity, and turn-rate for target 1
xPosTrue1 = xTargetCarTraj1[0,:][t1_startSample:t1_endSample+1] #ground truth X position
yPosTrue1 = xTargetCarTraj1[1,:][t1_startSample:t1_endSample+1] #ground truth Y position
xVelTrue1 = xTargetCarTraj1[2,:][t1_startSample:t1_endSample+1] #ground truth X velocity
yVelTrue1 = xTargetCarTraj1[3,:][t1_startSample:t1_endSample+1] #ground truth Y velocity
omegaTrue1 = xTargetCarTraj1[4,:][t1_startSample:t1_endSample+1] #ground truth turn-rate

#ground truth x position, y position, x velocity, y velocity, and turn-rate for target 2
xPosTrue2 = xTargetCarTraj2[0,:][t2_startSample:t2_endSample+1]
yPosTrue2 = xTargetCarTraj2[1,:][t2_startSample:t2_endSample+1]
xVelTrue2 = xTargetCarTraj2[2,:][t2_startSample:t2_endSample+1]
yVelTrue2 = xTargetCarTraj2[3,:][t2_startSample:t2_endSample+1]
omegaTrue2 = xTargetCarTraj2[4,:][t2_startSample:t2_endSample+1]

#initial ground truth states
x_01 = xTargetCarTraj1[:,0] 
x_02 = xTargetCarTraj2[:,0]

plt.figure()
plt.plot(xPosTrue1,yPosTrue1, color='b', label='True Trajectory Target 1')
plt.plot(xPosTrue2,yPosTrue2, color='r', label='True Trajectory Target 2')

plt.title("True Trajectory",fontname="Arial",fontsize=12)
plt.xlabel("X(m)",fontname="Arial",fontsize=12)
plt.ylabel("Y(m)",fontname="Arial",fontsize=12)
plt.legend()
plt.show()

#arrays below obtain the error between the estimated and actual states in each Monte Carlo run
errPosPlotsX1 = np.zeros((numRuns,N_t1)) #each row stores the estimation error for X position, each row corresponds to a Monte Carlo run
errPosPlotsY1 = np.zeros((numRuns,N_t1)) #each row stores the estimation error for Y position
errVelPlotsX1 = np.zeros((numRuns,N_t1))  #each row stores the estimation error for X velocity
errVelPlotsY1 = np.zeros((numRuns,N_t1)) #each row stores the estimation error for Y velocity
errOmegaPlots1 = np.zeros((numRuns,N_t1))  #each row stores the estimation error for turn-rate

errPosPlotsX2 = np.zeros((numRuns,N_t2))
errPosPlotsY2 = np.zeros((numRuns,N_t2))
errVelPlotsX2 = np.zeros((numRuns,N_t2))
errVelPlotsY2 = np.zeros((numRuns,N_t2))
errOmegaPlots2 = np.zeros((numRuns,N_t2))

RMSEs = np.zeros((n,numRuns)) #computes the scalar RMSE for each run

j=0
jj1 = 0
jj2 = 0

#in certain cases, there will likely be runs with track loss, hence, the number of these cases are counted 
#as one of the performance metrics
numErrCasesT1=0 
numErrCasesT2=0

#Tracking Loop
for ii in range(numRuns): #for each Monte Carlo run
    trackList = [0]*8000 #allocate track list, a list of objects
    lastTrackIdx = -1 #index of a track in the list,

    if t1_start == 0 or t2_start==0: #if the any target exists at the initial frame/sample, obtain their measurement
        if sensor=="Radar": #for radar case
            #xS = sensorPos[0]
            #yS = sensorPos[1]
            #xS = x0[0]+10
            #yS = x0[1]+10
            #x0_EgoCar = xEgoCarTraj[:,0]
            #sensorPos = np.array([x0_EgoCar[0],x0_EgoCar[1]])
            #sensorPos = np.array([xS,yS])
            
            if t1_start==0:
                r0= getRange(x0_Target1, sensorPos)
                angle0 = getAngle(x0_Target1, sensorPos)
                rangeMeas0 = r0 + np.random.normal(0,rangeStd,1) #obtain radar measurement
                angleMeas0 = angle0 + np.random.normal(0,angleStd,1)
            
            z0 = np.zeros((m,))
            z0[0] = rangeMeas0
            z0[1] = angleMeas0
            
            P_Post0 = np.zeros((n,n))
            
            lambda1 = math.exp(-(angleStd**2)/2)
            lambda2 = math.exp(-2*(angleStd**2))
            
            xS = sensorPos[0]
            yS = sensorPos[1]
            
            xPos0 = xS + (rangeMeas0/lambda1)*math.cos(angleMeas0)
            yPos0 = yS + (rangeMeas0/lambda1)*math.sin(angleMeas0)
            
            xPost0 = np.array([xPos0,yPos0,0,0,0,0,0])
            
            P_Post0[0,0] = (lambda1**(-2)  -2)*(rangeMeas0**2)*((math.cos(angleMeas0))**2) + 0.5*(rangeMeas0**2  + rangeStd**2)*(1+lambda2*math.cos(2*angleMeas0))
            P_Post0[1,1] =  (lambda1**(-2)  -2)*(rangeMeas0**2)*((math.sin(angleMeas0))**2) + 0.5*(rangeMeas0**2  + rangeStd**2)*(1-lambda2*math.cos(2*angleMeas0))
            P_Post0[0,1] =  (lambda1**(-2)  -2)*(rangeMeas0**2)*(math.cos(angleMeas0)*math.sin(angleMeas0)) + 0.5*(rangeMeas0**2  + rangeStd**2)*lambda2*math.sin(2*angleMeas0)
            P_Post0[1,0] = P_Post0[0,1]
            P_Post0[2,2] = (maxVel/2)**2
            P_Post0[3,3] = (maxVel/2)**2
            P_Post0[6,6] = (omegaMax/2)**2
            
            ePost0 = z0 - np.array([r0,angle0])
        else: #for Lidar case
            if m!=n: #if m<n, generate initial measurement of the target
                z0 = H@x0 + np.random.multivariate_normal(np.array([0,0]),R)
                #z0 = H@x0 + np.random.normal(0,sigma_w,2)
            else: #if m=n, which is when using S.A. Gadsden's SVSF
                z0 = H[0:2,0:2]@x0[0:2] + np.random.multivariate_normal(np.array([0,0]),R[0:2,0:2])
                z0 = np.array([z0[0],z0[1],0,0,0])
 
        track1  = track(z0,G,H,Q,R,maxVel,maxAcc,omegaMax, pInit,0,Ts,modelType,sensor,sensorPos, False,N) #initalize the track
        trackList[0] = track1
        lastTrackIdx=lastTrackIdx+1
        
    if sensor=="Lidar":
    
        #initial clutter points in the specified coverage regions
        xMin = x_01[0] - round(regionWidth/2) # x range of region around target 1, x_01[0] is the initial X position, x_01[1] is the initial Y position
        xMax = x_01[0] + round(regionWidth/2)
        
        yMin = x_01[1] - round(regionLength/2) #y range of region around target 2
        #xMax = x_01[0] + round(regionWidth/2)
        yMax = x_01[1] + round(regionLength/2)
        
        xLims1 = [xMin,xMax] #info for the coverage region around target 1
        yLims1 = [yMin,yMax]
        
        xMin = x_02[0] - round(regionWidth/2) #around target 2, x_02[0] is the initial X position, x_02[1] is the initial Y position
        xMax = x_02[0] + round(regionWidth/2)
        
        yMin = x_02[1] - round(regionLength/2)# y range of region around target 1
        yMax = x_02[1] + round(regionLength/2)
        
        xLims2 = [xMin,xMax] #info for the coverage region around target 2
        yLims2 = [yMin,yMax]
        
        clutterPoints1 = generateClutter(xLims1, yLims1, lambdaVal) #generate clutter around target 1
        clutterPoints2 = generateClutter(xLims2, yLims2, lambdaVal) #generate clutter around target 2
        unassignedMeas0 = np.hstack((clutterPoints1,clutterPoints2)) #full set of unassigned measurements
    else:
        clutterPoints = generateClutter(rangeLims, thetaLims, lambdaVal) #generate clutter around target 1
        #clutterPoints2 = generateClutter(rangeLims, thetaLims, lambdaVal) #generate clutter around target 2
        unassignedMeas0 = clutterPoints #full set of unassigned measurements
    
    #initial tracks, each unassigned measurement initiate a track
    trackList,lastTrackIdx = initiateTracks(trackList,lastTrackIdx, unassignedMeas0,  maxVel, maxAcc, omegaMax, G, H, Q, R, modelType, Ts, pInit, 0, sensor,sensorPos, N)

    #estTraj = np.zeros((n,N))
    #estTraj[:,0] = track1.xPost

    #zkPrev = z0    
    T1_Detected = 0
    T2_Detected = 0
    
    for k in range(1,N): #for each sample/frame/scan
        t_k = t[k] #current time in simulation
        T1_Present = (t_k>=t1_start and t_k<=t1_end) #check if target 1 is present
        T2_Present = (t_k>=t2_start and t_k<=t2_end) #check if target 2 is present
        
        T1_Detected = 0
        T2_Detected = 0   
        
        for i in range(numTargets):
            if  (T1_Present and i==0) or (T2_Present and i==1): #if any target is present
                u = random.uniform(0, 1) #generate uniform random number 
                if u<=PD:  #if the target is detected obtain its measurement    
                    if i==0: #if its target 1
                        x_k = xTargetCarTraj1[:,k]
                        T1_Detected = 1
                    elif i==1: #if its target 2
                        x_k= xTargetCarTraj2[:,k]
                        T2_Detected = 1
    
                    if sensor=="Radar":
#                        xk_EgoCar = xEgoCarTraj[:,k]
                        rangeMeas = getRange(x_k, sensorPos) + np.random.normal(0,rangeStd,1) #obtain radar measurement
                        angleMeas = getAngle(x_k, sensorPos) + np.random.normal(0,angleStd,1)
                        
                        z_k = np.zeros((m,))
                        z_k[0] = rangeMeas
                        z_k[1] = angleMeas
                    else:
                        
                        #for lidar
                        if m==n: #for using S.A. Gadsden's SVSF
                            z_kPos = H[0:2,0:2]@x_k[0:2] + np.random.normal(0,sigma_w,2)
                            vx_m = (z_kPos[0] - zkPrev[0])/Ts #generate artifical measurements
                            vy_m = (z_kPos[1] - zkPrev[1])/Ts
                            omega_m = 0
                            
                            z_k = np.zeros((n,))
                            z_k[0] = z_kPos[0]
                            z_k[1] = z_kPos[1]
                            z_k[2] = vx_m
                            z_k[3] = vy_m
                            z_k[4] = omega_m
                        else:
                            #If using Mina's SVSF
                            
                            #z_k = H@x_k +  np.random.multivariate_normal(np.array([0,0]),R) #recieve measurement
                            z_k = H@x_k + np.random.normal(0,sigma_w,2)
                            
                    if i==0 :
                        z_k1 = z_k #if target 1, store z_k into z_k1
                    else:
                        z_k2 = z_k #if target 2, store z_k into z_k2
                else:
                   # z_k =  np.array([math.inf,math.inf])#if no target is detected
                    if i==0:
                        T1_Detected=0
                    elif i==1:
                        T2_Detected=0
                    
        if T1_Present==True and T2_Present==False:
            if T1_Detected==0:
                targetMeas = [] #if target 1 is present but not detected, make the targetMeas array empty
            else:
                targetMeas = np.expand_dims(z_k1,axis=-1) #if the target is present and detected obtain store it in targetMeas
        elif T1_Present==False and T2_Present==True:
            if T2_Detected==0:
                targetMeas = [] #if target 2 is present but not detected, make the array empty
            else:
                targetMeas = np.expand_dims(z_k2,axis=-1) #if present and detected, store its measurement into targetMeas
        elif T1_Present==True and T2_Present==True: 
            if T1_Detected==1 and T2_Detected==1: #if both targets are present and detected, store their measurements into targetMeas
                targetMeas = np.hstack((np.expand_dims(z_k2,axis=-1) ,np.expand_dims(z_k1,axis=-1) ))
            elif T1_Detected==1 and T2_Detected==0 :
                targetMeas = np.expand_dims(z_k1,axis=-1) #if both targets are present, but only target 1 is detected 
            elif T1_Detected==0 and T2_Detected==1:
                targetMeas = np.expand_dims(z_k2,axis=-1) #if both targets are present, but only target 2 is detected
            else:
                targetMeas = []
        else:
            targetMeas = []
                            
        if filterType == "PDAKF" or filterType == "IPDAKF" or filterType=="IPDASVSF" or  filterType == "IPDAGVBLSVSF":
            #Generate clutter     
            if sensor=="Lidar":
                for i in range(numTargets):
                    if i==0:
                        x_k = xTargetCarTraj1[:,k] #true state of target 1
                        #z_k = z_k1
                    elif i==1:
                        x_k = xTargetCarTraj2[:,k] #true state of target 2
                        #z_k = z_k2
                
                
                    sensorPos = np.array([x_k[0],x_k[1]]) #for radar, set at the same position of the target
                    #generate region, xMin is the min. x in the region, xMax is the max x. in the region, yMin is the min. y, and yMax is the max. y in the region
                    #x_k[0] and x_k[1] denote the X and Y position of the target
                    xMin = x_k[0] - round(regionWidth/2) 
                    xMax = x_k[0] + round(regionWidth/2)
                
                    yMin = x_k[1] - round(regionLength/2)
                    yMax = x_k[1] + round(regionLength/2)
                    
                    xLims = [xMin,xMax]
                    yLims = [yMin,yMax]
                    
                    clutterPoints = generateClutter(xLims, yLims, lambdaVal) #generate clutter points
                    if i==0:
                        clutterPoints1 = clutterPoints #clutter points around target 1
                    else:
                        clutterPoints2 = clutterPoints #clutter points around target 2
            
                if isinstance(targetMeas, list)==False: #if a target is detected
                    measSet = np.hstack((clutterPoints1,clutterPoints2,targetMeas)) #full set of measurements
                else: #if no target is detected
                    measSet = np.hstack((clutterPoints1,clutterPoints2)) #full set of measurements
            elif sensor=="Radar":
                clutterPoints = generateClutter(rangeLims, thetaLims, lambdaVal) #generate clutter points
                
                if isinstance(targetMeas, list)==False: #if a target is detected
                    measSet = np.hstack((clutterPoints,targetMeas)) #full set of measurements
                else: #if no target is detected
                    measSet = clutterPoints #full set of measurements
                
            #perform gating
            trackList,unassignedMeas, measSet = gating(trackList,lastTrackIdx,PG, MP, maxVals,sensorPos,measSet,k) #perform gating
            trackList = updateStateTracks(trackList,lastTrackIdx, filterType, measSet,maxVals, lambdaVal,MP, PG, PD, sensorPos,SVSFParams, k) #update the state of each track
            
            if predTraj == True: #predict trajectory for each track
                trackList = predTrajsTracks(trackList, lastTrackIdx, tP)
            
            if useLogic==True:
                trackList = updateTracksStatus_MN(trackList,lastTrackIdx,N1,M2,N2,N3,k)
            else:
                trackList = updateTracksStatus(trackList,lastTrackIdx, delTenThr, delConfThr, confTenThr,k) #update the status of each track usiing the track manager
            
            #initiate tracks for measurements that were not gated or in other words unassigned measurements
            trackList,lastTrackIdx = initiateTracks(trackList,lastTrackIdx,unassignedMeas,  maxVel, maxAcc, omegaMax, G, H, Q, R, modelType, Ts, pInit, k, sensor,sensorPos,  N)
   
    numTracks = lastTrackIdx+1 #total number of tracks
    
    for i in range(numTracks): #loop through the track list
        status = trackList[i].status
        if status==1 or status==2: #if the track is tenative or confirmed, set the endSample property to the last sample, which is the value of k at the end of the loop above
            trackList[i].endSample = k
    
    for j in range(numTracks): #loop through each track
       # isTrack1 = abs(t1_startSample - trackList[j].startSample) <10 #Boolean value, associate track to target 1 if the startSample is close to the true start sample
        #isTrack2 = abs(t2_startSample - trackList[j].startSample) <10 #Boolean value, associate track to target 2 if the startSample is close to the true start sample
        
        #numSamples = trackList[j].endSample - trackList[j].startSample + 1 #total number of samples processed in the track
        
       #if (isTrack1==True or isTrack2==True) and numSamples>10: #if the track is associated to one of the targets and the number of samples is greater than 100
                                                                #obtain its state estimates
        xEsts = trackList[j].xEsts
            #if isTrack1==True: #if a track is associated to target 1
        xPosEst_1 = xEsts[0,:][t1_startSample:t1_endSample+1] #Obtain estimated positions and velocities
        yPosEst_1 = xEsts[1,:][t1_startSample:t1_endSample+1] 
        xVelEst_1 = xEsts[2,:][t1_startSample:t1_endSample+1] 
        yVelEst_1 = xEsts[3,:][t1_startSample:t1_endSample+1] 
        
        xPosRMSE_1 = computeRMSE(xPosTrue1, xPosEst_1) #get RMSE
        yPosRMSE_1 = computeRMSE(yPosTrue1, yPosEst_1)
        
        
        if numTargets==2:
            xPosEst_2 = xEsts[0,:][t2_startSample:t2_endSample+1] #Obtain estimated positions and velocities
            yPosEst_2 = xEsts[1,:][t2_startSample:t2_endSample+1] 
            xVelEst_2 = xEsts[2,:][t2_startSample:t2_endSample+1] 
            yVelEst_2 = xEsts[3,:][t2_startSample:t2_endSample+1] 
            
            xPosRMSE_2 = computeRMSE(xPosTrue2, xPosEst_2) #get RMSE
            yPosRMSE_2 = computeRMSE(yPosTrue2, yPosEst_2)
            
        BLs_T1 = trackList[j].BLs
                
        if xPosRMSE_1 > errThr or yPosRMSE_1 > errThr:
            numErrCasesT1 = numErrCasesT1+1 #ignore runs with high error~
            '''
            if jj1<errPosPlotsX1.shape[0]:
                np.delete(errPosPlotsX1,jj1,axis=0)
                np.delete(errPosPlotsY1,jj1,axis=0)
                np.delete(errVelPlotsX1,jj1,axis=0)
                np.delete(errVelPlotsY1,jj1,axis=0)
                np.delete(errOmegaPlots1,jj1,axis=0)  
            '''
        else:
            xPosErr = xPosEst_1-xPosTrue1 #find error in each state
            yPosErr = yPosEst_1-yPosTrue1
            xVelErr = xVelEst_1-xVelTrue1
            yVelErr = yVelEst_1-yVelTrue1
            errPosPlotsX1[jj1,:] = xPosErr #store errors in the array containing the errors from each run
            errPosPlotsY1[jj1,:] = yPosErr
            errVelPlotsX1[jj1,:] = xVelErr
            errVelPlotsY1[jj1,:] = yVelErr
            jj1=jj1+1
            
        if numTargets==2:
            if xPosRMSE_2 > errThr2 or yPosRMSE_2 >errThr2:
                numErrCasesT2 = numErrCasesT2+1 #ignore runs with high error~
                '''
                if jj2<errPosPlotsX1.shape[0]:
                    np.delete(errPosPlotsX2,jj2,axis=0)
                    np.delete(errPosPlotsY2,jj2,axis=0)
                    np.delete(errVelPlotsX2,jj2,axis=0)
                    np.delete(errVelPlotsY2,jj2,axis=0)
                    np.delete(errOmegaPlots2,jj2,axis=0)
                '''
            else:
                BLs_T2 = trackList[j].BLs

                xPosErr = xPosEst_2-xPosTrue2#find error in each state
                yPosErr = yPosEst_2-yPosTrue2
                xVelErr = xVelEst_2-xVelTrue2
                yVelErr = yVelEst_2-yVelTrue2
                
                #if xPosRMSE < errThr:
                errPosPlotsX2[jj2,:] = xPosErr#Obtain estimated positions and velocities
                errPosPlotsY2[jj2,:] = yPosErr
                errVelPlotsX2[jj2,:] = xVelErr
                errVelPlotsY2[jj2,:] = yVelErr
                jj2=jj2+1
        
        
#        estTraj[:,k]=xPost
 #       zkPrev = z_k
    """
    xPosRMSE = computeRMSE(xPosTrue, xPosEst)
    yPosRMSE = computeRMSE(yPosTrue, yPosEst)
    xVelRMSE = computeRMSE(xVelTrue, xVelEst) 
    yVelRMSE = computeRMSE(yVelTrue, yVelEst)
    omegaRMSE = computeRMSE(omegaTrue, omegaEst)
    
    rmseVec = np.array([xPosRMSE,yPosRMSE, xVelRMSE, yVelRMSE, omegaRMSE])

    errPosX = xPosEst-xPosTrue #Obtain errors
    errPosY = yPosEst-yPosTrue
    errVelX = xVelEst-xVelTrue
    errVelY = yVelEst-yVelTrue
    errOmega = omegaEst - omegaTrue
    
    if xPosRMSE > errThr or yPosRMSE > errThr:
        numErrCases = numErrCases+1 #ignore runs with high error
        np.delete(errPosPlotsX,j,axis=0)
        np.delete(errPosPlotsY,j,axis=0)
        np.delete(errVelPlotsX,j,axis=0)
        np.delete(errVelPlotsY,j,axis=0)
        np.delete(errOmegaPlots,j,axis=0)
    else:
        errPosPlotsX[j,:] = errPosX #store errors
        errPosPlotsY[j,:] = errPosY
        errVelPlotsX[j,:] = errVelX
        errVelPlotsY[j,:] = errVelY
        errOmegaPlots[j,:] = errOmega
        
        j=j+1
    
    RMSEs[:,i] = rmseVec
    """
        
if errPosPlotsX1.sum()!= 0:
    rmsePlotPos1,xPosRMSE1,yPosRMSE1,posRMSE1 = RMSEPlot(errPosPlotsX1,errPosPlotsY1)
    rmsePlotVel1,xVelRMSE1,yVelRMSE1,velRMSE1 = RMSEPlot(errVelPlotsX1,errVelPlotsY1)
    
    plt.figure()
    plt.plot(t[t1_startSample:t1_endSample+1],rmsePlotPos1)
    plt.title("Position RMSE Over-time for Target 1")
    plt.xlabel("Time(s)")
    plt.ylabel("RMSE(m)")
    plt.show()

    plt.figure()
    plt.plot(t[t1_startSample:t1_endSample+1],rmsePlotVel1)
    plt.title("Velocity RMSE Over-time for Target 1")
    plt.xlabel("Time(s)")
    plt.ylabel("RMSE(m/s)")
    plt.show()

    if modelType == "CT":
        rmsePlotOmega1,omegaRMSE1,_,_ = RMSEPlot(errOmegaPlots1,np.zeros((numRuns,N)))
        
        plt.figure()
        plt.plot(t[t1_startSample:t1_endSample+1],rmsePlotOmega1)
        plt.title("Turn-rate RMSE Over-time for Target 1")
        plt.xlabel("Time(s)")
        plt.ylabel("RMSE(rad/s)")
        plt.show()

if numTargets==2:    
    if errPosPlotsX2.sum()!= 0:
        rmsePlotPos2,xPosRMSE2,yPosRMSE2,posRMSE2 = RMSEPlot(errPosPlotsX2,errPosPlotsY2)
        rmsePlotVel2,xVelRMSE2,yVelRMSE2,velRMSE2 = RMSEPlot(errVelPlotsX2,errVelPlotsY2)
        
        plt.figure()
        plt.plot(t[t2_startSample:t2_endSample+1],rmsePlotPos2)
        plt.title("Position RMSE Over-time for Target 2")
        plt.xlabel("Time(s)")
        plt.ylabel("RMSE(m)")
        plt.show()
        
        plt.figure()
        plt.plot(t[t2_startSample:t2_endSample+1],rmsePlotVel2)
        plt.title("Velocity RMSE Over-time for Target 2")
        plt.xlabel("Time(s)")
        plt.ylabel("RMSE(m/s)")
        plt.show()
    
        if modelType == "CT":
            rmsePlotOmega2,omegaRMSE2,_,_ = RMSEPlot(errOmegaPlots2,np.zeros((numRuns,N)))
            plt.figure()
            plt.plot(t[t2_startSample:t2_endSample+1],rmsePlotOmega2)
            plt.title("Turn-rate RMSE Over-time for Target 2")
            plt.xlabel("Time(s)")
            plt.ylabel("RMSE(rad/s)")
            plt.show()

numTracks = lastTrackIdx + 1
for i in range(numTracks):
    status = trackList[i].status
    if status==1 or status==2:
        trackList[i].endSample = k
        
numSamplesArr = np.zeros((numTracks,1))
plt.figure()

for i in range(numTracks): #for each track
    status = trackList[i].status
    numSamples = trackList[i].endSample - trackList[i].startSample+ 1 #obtain the number of processed samples
     
    numSamplesArr[i] = numSamples
    if numSamples > 20: #if the number of samples processed is above 10
        xEsts = trackList[i].xEsts #obtain state estimates
        startSample = trackList[i].startSample
        endSample = trackList[i].endSample
        xPosEst = xEsts[0,:] #Obtain estimated positions and velocities
        yPosEst = xEsts[1,:]
        plt.plot(xPosEst[startSample:endSample],yPosEst[startSample:endSample]) #plot estimated trajectory
        plt.show()

print(datetime.now() - startTime)

if errPosPlotsX1.sum()!=0:
    #print("The number of error cases for Target 1 are:" + str(numErrCasesT1))
    print("xPosRMSE="+str(xPosRMSE1))
    print("yPosRMSE="+ str(yPosRMSE1))
    print("xVelRMSE="+str(xVelRMSE1))
    print("yVelRMSE="+str(yVelRMSE1))

    if modelType == "CT":
        print("omegaRMSE="+str(omegaRMSE1))
print("\n")
#print("The number of error cases for Target 2 are:" + str(numErrCasesT2))

if numTargets == 2 and errPosPlotsX2.sum()!=0:
    print("xPosRMSE="+str(xPosRMSE2))
    print("yPosRMSE="+ str(yPosRMSE2))
    print("xVelRMSE="+str(xVelRMSE2))
    print("yVelRMSE="+str(yVelRMSE2))
    
    if modelType == "CT":
        print("omegaRMSE="+str(omegaRMSE2))

if filterType == "IPDASVSF":
    xPosBLs = BLs_T1[0,:]
    yPosBLs = BLs_T1[1,:]
    xVelBLs = BLs_T1[2,:]
    yVelBLs = BLs_T1[3,:]
    omegaBLs= BLs_T1[4,:]
    
