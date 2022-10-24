import numpy as np




def get_features(pc, bbox):
    n = pc.shape[0]  # num
    l = bbox[3]  # L
    w = bbox[4]  # W
    h = bbox[5]  # S
    if l != 0 and w != 0 and h != 0:
        d = n / l / w / h  # density
    else:
        d = 0
    stdx = np.std(pc[:, 0])  # stdx
    stdy = np.std(pc[:, 1])  # stdx
    stdz = np.std(pc[:, 2])  # stdz
    stdv = np.std(pc[:, 3])  # stdv
    vmean = pc[:, 3].mean()  # v mean
    vrange = pc[:, 3].max() - pc[:, 3].min()  # v range
    rg = np.sqrt(np.sum(np.square([bbox[0], bbox[1], bbox[2]])))

    return n, l, w, h, d, rg, stdx, stdy, stdz, stdv, vmean, vrange


def distance_weighted_voting(dets, dis):
    l = [0, 0, 0, 0]
    d = np.square(dis[:, :2])
    d = np.sqrt(np.sum(d, axis=1))
    minimum = d[np.argmax(d)]
    maximum  = d[np.argmin(d)]

    for i in range(len(dets)):
        rg = np.square(dis[i, :2])
        rg = np.sqrt(np.sum(rg))
        score = 1-((rg-minimum)/(maximum-minimum))/2
        l[dets[i]] += score
    return l.index(max(l))


def distance_weighted_voting_custom(dets, dis, weights):
    l = [0, 0, 0, 0]
    for i in range(len(dets)):
        rg = np.square(dis[i, :2])
        rg = np.sqrt(np.sum(rg))
        p = 0
        weight_index = len(weights)
        for ii, j in enumerate(range(10, 100, 10)):
            if rg > p and rg < j:
                weight_index = ii
                p = j

        if weight_index >= weights.shape[0]:
            weight_index = weights.shape[0]-1
        weight = weights[weight_index]
        score = (1-weight)*dets[i, :]
        l += score
    l = list(l)
    return [l.index(max(l))]



if __name__ == "__main__":
    """
[1, 2, 8, 10, 12, 18, 21, 25, 30, 39, 54, 63, 68, 70, 71, 72, 73, 75, 76, 77, 79, 80, 81, 82, 83, 86, 88]
[8, 9, 11, 1, 13, 18, 19, 21, 25, 27, 10, 12, 35, 46, 49, 36, 63, 54, 70, 71, 72, 73, 75, 76, 77, 79, 82, 81, 87, 45, 83]
    rad_id 10 is camid 8
    Assumptions
        we are only detecting object in black region
        3 zones object should only enter or exit in zones 1 (red) or 2 (green).
        zone 0 is everywhere not zone 1 or zone 2
        If object entered in zone 1, object must exit in zone 2 (vice versa)
    
    Tracking routine
    Zone_state = [-1]*8000
    Track j is initialized:
        if track[j].first_location in zone_1:
            zone_state[j] = zone_1
        if track[j].first_location in zone_2:
            zone_state[j] = zone_2
        else:
            zone_state[j] = 0
    Track j is terminated:
        if zone_state[j] == 1:
            if track[j].final_location in zone 1:
                ?
            if track[j].final_location in zone 0:
                track[j] entered in the right place but exited in the wrong place
            if track[j].final_location in zone 2:
                track[j] is correct
        
        if zone_state[j] == 2:
            if track[j].final_location in zone 2:
                ?
            if track[j].final_location in zone 0:
                track[j] entered in the right place but exited in the wrong place
            if track[j].final_location in zone 1:
                track[j] is correct
                
        if zone_state[j] == 0:
            if track[j].final_location in zone 1 or in zone 2:
                track[j] entered in the wrong place but exited in the right place
            if track[j].final_location in zone 0:
                track[j] entered and exited in the wrong place 

                
        
    in total there are 5 cases
        0. track[j] is correct
        1. track[j] entered in the right place but exited in the wrong place
        2. track[j] entered in the wrong place but exited in the right place
`       3. track[j] entered and exited in the wrong place 
        4. ? is track has appeared and disappeared in the same zone
            here the track could be travelling from zone 1 to zone 2 or zone 2 to zone 1
            we do know it entered or exited in the correct zone 
        
        
        
        Following the assumptions
        case 0
            All good
        case 1
            The track must have not finished, a future track must be a continuation of this track
        case 2
            The track must resulted from a track which was not finished
        case 3
            This track must have resulted from a not finished track and a future track must be a continuation of it
        case 4
            This track can be a continuation of a not finished track or a track which was not finished
    
    
    Some more ideas we can further sub divide the zone to include direction I guess? 
    But there could be more error this way
    
    
    
ID 5, case='case_4'
ID 0, case='case_4'
ID 6, case='case_4'
ID 2, case='case_2'





    """
    import matplotlib.pyplot as plt
    trig = np.array([[    -2.4517,      31.843,     0.40313,    -0.93226],
       [    -2.2685,      32.226,     0.68361,     -1.0643],
       [    -1.4016,       32.22,    0.046236,      -1.381],
       [    -1.2853,      32.144,     0.12782,     -1.2438],
       [    -2.0725,      31.039,    0.079475,    -0.93483],
       [    -2.7309,      30.407,    -0.15347,    -0.85092],
       [    -2.8608,      30.172,    0.089997,    -0.80718],
       [    -2.5181,      30.622,      0.2056,    -0.72269],
       [    -3.1895,      29.811,  2.1471e-05,    -0.67859],
       [    -2.3686,      30.398,    0.056576,    -0.67476],
       [     -3.787,      28.864,   -0.055058,     -0.6624],
       [    -2.5919,      30.308,   -0.062053,    -0.69323],
       [    -2.7433,      30.002,    -0.17762,    -0.51456],
       [    -2.8958,      29.808,    0.050603,    -0.58913],
       [    -2.2846,      30.472,    -0.17997,    -0.62091],
       [    -3.7915,      27.836,   -0.027876,    -0.70596],
       [    -1.9393,      31.146,   -0.055855,    -0.76286],
       [    -2.5751,      30.157,    0.016016,     -0.6357],
       [     -3.412,      28.918,    0.037589,    -0.81796],
       [      -3.43,      29.405,   -0.026234,    -0.84403],
       [     -4.019,       28.39,    -0.06194,     -1.0598],
       [    -4.4915,      28.074,    -0.11119,     -0.8145],
       [    -3.8473,      28.343,     -0.1915,     -1.0564],
       [    -3.4673,      29.123,   -0.047436,    -0.91264],
       [    -3.8764,      28.061,    0.016389,    -0.72949],
       [    -4.6382,        27.7,    -0.40384,    -0.85824],
       [    -3.4877,       28.89,    0.043196,    -0.97675],
       [     -3.432,      29.024,     0.05748,    -0.91509],
       [    -4.7732,      27.539,    -0.16729,    -0.75072],
       [    -3.2473,      29.372,   -0.014852,    -0.82094],
       [    -3.8786,      28.243,    -0.16368,    -0.79626],
       [    -3.7964,      28.449,    -0.22289,    -0.54954],
       [    -5.8897,      24.645,    -0.16839,    -0.49055],
       [    -4.3003,      28.145,    -0.11641,    -0.63116],
       [    -4.1684,      28.023,    -0.10159,    -0.58696],
       [    -4.6957,      26.572,     -0.2014,    -0.56467],
       [    -5.2951,      26.891,    -0.31698,     -0.4416],
       [    -4.7925,      27.118,    -0.26542,    -0.45396],
       [    -5.4702,      26.489,    -0.66864,     -0.4416],
       [    -4.7461,      26.923,    -0.31617,    -0.55568],
       [    -4.9874,       26.71,    -0.32735,    -0.55276],
       [    -5.1246,      26.604,    -0.24954,    -0.60265],
       [    -4.7185,      27.536,     0.12677,    -0.58077],
       [    -4.4297,      27.627,   -0.037804,    -0.65163],
       [    -4.8697,      27.248,    -0.10782,    -0.73549],
       [    -4.8138,      27.383,   -0.078745,    -0.74718],
       [    -6.3323,      24.889,   -0.025641,    -0.53857],
       [    -4.3741,       27.21,    -0.19863,    -0.84566],
       [    -5.4011,      26.001,    -0.41822,    -0.78857],
       [    -5.1592,      26.079,    -0.31535,    -0.80912],
       [    -4.0324,      27.829,   -0.083353,    -0.74441],
       [    -4.8087,      26.563,   -0.060862,     -0.7008],
       [    -4.6782,      26.983,   -0.091661,    -0.69845],
       [    -4.5118,      27.156,     0.13323,    -0.60301],
       [     -4.192,      27.488,    -0.16909,    -0.57997],
       [    -4.4192,      27.225,   -0.099734,    -0.58039],
       [    -4.4868,      26.932,    -0.15633,    -0.52992],
       [     -4.955,       26.39,    -0.17593,    -0.49875],
       [    -4.8325,      26.413,   -0.086406,    -0.46286],
       [    -4.3182,       26.96,     0.19998,    -0.44896],
       [    -4.4519,       26.97,    -0.11777,     -0.4364],
       [    -4.8685,      26.979,     0.21477,    -0.38941],
       [    -4.8233,      27.216,    -0.21433,    -0.37977],
       [    -4.0524,      27.642,    -0.37684,    -0.35529],
       [    -4.3814,      27.225,    -0.15036,    -0.29532],
       [    -4.3064,      27.091,    -0.30707,    -0.19784],
       [    -4.6389,      26.797,    -0.22709,    -0.12551],
       [   -0.93853,      30.125,     0.02813,    -0.57408],
       [   -0.75586,      30.409,     0.11969,    -0.52992],
       [   -0.96114,      30.102,     -0.1225,    -0.39744]])




    labels = np.array([[    0.91738,    0.027737,   0.0012855,    0.053596],
       [    0.81397,    0.048052,   0.0081183,     0.12986],
       [    0.73342,    0.037427,    0.026925,     0.20223],
       [    0.73156,    0.027026,    0.029233,     0.21218],
       [       0.75,   0.0077236,   0.0001227,     0.24216],
       [    0.52035,    0.024432,   9.786e-05,     0.45512],
       [    0.81421,    0.073659,    0.001613,     0.11052],
       [    0.78664,    0.065523,   0.0010863,     0.14675],
       [    0.60062,    0.026723,  0.00052646,     0.37214],
       [    0.87122,    0.021844,   0.0009512,     0.10598],
       [    0.73739,     0.11296,    0.052444,    0.097201],
       [    0.90273,    0.010626,   0.0015235,     0.08512],
       [    0.80653,    0.014864,   0.0051016,      0.1735],
       [    0.83332,    0.018418,   0.0012814,     0.14698],
       [    0.92657,    0.016866,   0.0041411,     0.05242],
       [   0.062048,    0.016467,  0.00023468,     0.92125],
       [    0.88465,    0.018853,    0.008562,    0.087933],
       [     0.8473,    0.016041,   0.0013461,     0.13531],
       [    0.56693,    0.085786,  0.00070708,     0.34657],
       [    0.93884,   0.0090218,  0.00031091,    0.051826],
       [    0.79821,    0.039697,      0.0222,     0.13989],
       [    0.87922,    0.045918,    0.013776,    0.061084],
       [    0.80561,    0.043543,    0.020552,      0.1303],
       [     0.9491,   0.0084798,   0.0010733,    0.041349],
       [    0.61533,    0.010787,  0.00038823,     0.37349],
       [    0.79984,    0.062294,    0.035415,     0.10245],
       [    0.83183,    0.018921,  0.00054861,      0.1487],
       [    0.92925,    0.014698,  0.00071974,     0.05533],
       [    0.78335,    0.096147,    0.034035,    0.086465],
       [    0.91901,    0.024087,  0.00055898,    0.056343],
       [    0.88205,    0.019113,   0.0028094,    0.096029],
       [     0.9518,   0.0085688,  0.00076175,    0.038872],
       [   0.092585,    0.033725,  0.00061906,     0.87307],
       [    0.86758,    0.030821,   0.0065873,    0.095007],
       [    0.77006,    0.069908,   0.0070785,     0.15296],
       [    0.28575,    0.045977,   0.0011685,      0.6671],
       [    0.74791,     0.10474,    0.063984,    0.083368],
       [    0.79401,    0.081606,    0.031925,    0.092454],
       [     0.7513,    0.092782,    0.061036,    0.094886],
       [     0.7763,    0.053933,   0.0058739,     0.16389],
       [    0.86944,     0.04106,    0.017801,    0.071696],
       [    0.88313,    0.020065,   0.0090431,    0.087763],
       [     0.8265,    0.072039,   0.0093361,    0.092121],
       [    0.91998,    0.015061,   0.0013436,     0.06361],
       [    0.91387,    0.023329,   0.0019497,    0.060855],
       [    0.93002,     0.02274,   0.0014829,    0.045759],
       [     0.3111,    0.009429,  7.4373e-05,      0.6794],
       [    0.86542,   0.0095303,  6.2992e-05,     0.12498],
       [    0.85873,    0.038338,    0.020632,    0.082297],
       [    0.82235,    0.027288,    0.018061,      0.1323],
       [     0.9253,    0.030897,    0.002108,    0.041695],
       [    0.91137,    0.022749,   0.0015746,    0.064306],
       [    0.91802,    0.016552,   0.0020353,    0.063391],
       [    0.92286,    0.024826,  0.00085728,    0.051458],
       [    0.93353,    0.011934,   0.0015122,    0.053028],
       [    0.86806,   0.0058289,   3.367e-05,     0.12608],
       [    0.94082,     0.01024,   0.0020857,    0.046857],
       [    0.87857,    0.019301,  0.00075262,     0.10138],
       [    0.87387,     0.04104,     0.01064,    0.074449],
       [    0.89364,    0.035643,    0.014242,    0.056471],
       [    0.88842,    0.024566,   0.0089968,    0.078014],
       [    0.92076,    0.017764,   0.0023564,     0.05912],
       [    0.94953,    0.014081,   0.0012734,    0.035115],
       [    0.79774,    0.037045,     0.04796,     0.11726],
       [    0.90797,    0.021624,   0.0060704,    0.064339],
       [    0.91852,    0.027114,    0.010228,    0.044136],
       [    0.91812,    0.029135,   0.0076333,    0.045107],
       [    0.79208,    0.081215,    0.030575,    0.096126],
       [     0.7683,     0.11408,    0.001805,     0.11581],
       [    0.81629,    0.069846,    0.021014,    0.092846]])


    d_weight = np.array([1.25998060e-01, 0.00000000e+00, 3.72307987e-04, 2.97648267e-04,
       4.07682852e-03, 3.03747762e-03, 2.07011840e-02, 2.03772601e-01,
       5.00000000e-01])
    x = trig[:, 0]
    y = trig[:, 1]
    # label = distance_weighted_voting(labels, trig)
    label = distance_weighted_voting_custom(labels, trig, d_weight)
    # print(label)
    # plt.plot(x, y, mew=0)
    # plt.show()