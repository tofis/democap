def get_markers():
    MARKERS = {}

    MARKERS["01"] = (4, 106, 1+ 100)     # 00 spinebase
    MARKERS["02"] = (8, 104, 2+ 100)     # 01 left chest
    MARKERS["03"] = (12, 102, 3+ 100)   # 02 right chest
    MARKERS["04"] = (16, 100, 4+ 100)   # 03 left head
    MARKERS["05"] = (20, 98, 5+ 100)   # 04 right head
    MARKERS["06"] = (24, 96, 6+ 100)   # 05 back_head
    MARKERS["07"] = (28, 94, 7+ 100)   # 06 back_high
    MARKERS["08"] = (32, 92, 8+ 100)   # 07 back_low
    MARKERS["09"] = (36, 90, 9+ 100)   # 08 left b_shoulder
    MARKERS["10"] = (40, 88, 10+ 100)   # 09 left f_shoulder
    MARKERS["11"] = (44, 86, 11+ 100)   # 10 left upperarm
    MARKERS["12"] = (48, 84, 12+ 100)   # 11 left forearm
    MARKERS["13"] = (52, 82, 13+ 100)   # 12 right b_shoulder
    MARKERS["14"] = (56, 80, 14+ 100)   # 13 right f_shoulder
    MARKERS["15"] = (60, 78, 15+ 100)   # 14 right upperarm
    MARKERS["16"] = (64, 76, 16+ 100)   # 15 right forearm   
    MARKERS["17"] = (68, 74, 17+ 100)   # 16 left pelvis
    MARKERS["18"] = (72, 72, 18+ 100)   # 17 left thigh
    MARKERS["19"] = (76, 70, 19+ 100)   # 18 left calf
    MARKERS["20"] = (80, 68, 20+ 100)   # 19 right pelvis
    MARKERS["21"] = (84, 66, 21+ 100)    # 20 right thigh
    MARKERS["22"] = (88, 64, 22+ 100)    # 21 right calf

    MARKERS["23"] = (92, 62, 23+ 100)   # 22 left hand
    MARKERS["24"] = (96, 60, 24+ 100)   # 23 left foot

    MARKERS["25"] = (100, 58, 25+ 100)   # 24 right hand
    MARKERS["26"] = (104, 56, 26+ 100)    # 25 right foot

    MARKERS["27"] = (108, 54, 27+ 100)   # 22 left hand
    MARKERS["28"] = (112, 52, 28+ 100)   # 23 left foot

    MARKERS["29"] = (116, 50, 29+ 100)   # 24 right hand
    MARKERS["30"] = (120, 48, 30+ 100)    # 25 right foot

    MARKERS["31"] = (124, 46, 31+ 100)     # 00 spinebase
    MARKERS["32"] = (128, 44, 32+ 100)     # 01 left chest
    MARKERS["33"] = (132, 42, 33+ 100)   # 02 right chest
    MARKERS["34"] = (136, 40, 34+ 100)   # 03 left head
    MARKERS["35"] = (140, 38, 35+ 100)   # 04 right head
    MARKERS["36"] = (144, 36, 36+ 100)   # 05 back_head
    MARKERS["37"] = (148, 34, 37+ 100)   # 06 back_high
    MARKERS["38"] = (152, 32, 38+ 100)   # 07 back_low
    MARKERS["39"] = (156, 30, 39+ 100)   # 08 left b_shoulder
    MARKERS["40"] = (160, 28, 40+ 100)   # 09 left f_shoulder
    MARKERS["41"] = (164, 26, 41+ 100)   # 10 left upperarm
    MARKERS["42"] = (168, 24, 42+ 100)   # 11 left forearm
    MARKERS["43"] = (172, 22, 43+ 100)   # 12 right b_shoulder
    MARKERS["44"] = (176, 20, 44+ 100)   # 13 right f_shoulder
    MARKERS["45"] = (180, 18, 45+ 100)   # 14 right upperarm
    MARKERS["46"] = (184, 16, 46+ 100)   # 15 right forearm   
    MARKERS["47"] = (188, 14, 47+ 100)   # 16 left pelvis
    MARKERS["48"] = (192, 12, 48+ 100)   # 17 left thigh
    MARKERS["49"] = (196, 10, 49+ 100)   # 18 left calf
    MARKERS["50"] = (200, 8, 50+ 100)   # 19 right pelvis
    MARKERS["51"] = (204, 6, 51+ 100)    # 20 right thigh
    MARKERS["52"] = (208, 4, 52+ 100)    # 21 right calf
    MARKERS["53"] = (216, 2, 53+ 100)    # 21 right calf

    


    return MARKERS

def get_markers_deepmocap():
    MARKERS = {}

    MARKERS["01"] = (0, 255, 0+ 100)     # 00 spinebase
    MARKERS["02"] = (255, 0, 0+ 100)     # 01 left chest
    MARKERS["03"] = (255, 255, 0+ 100)   # 02 right chest
    MARKERS["04"] = (0, 255, 255+ 100)   # 03 left head
    MARKERS["05"] = (255, 0, 255+ 100)   # 04 right head
    MARKERS["06"] = (185, 255, 0+ 100)   # 05 back_head
    MARKERS["07"] = (0, 185, 255+ 100)   # 06 back_high
    MARKERS["08"] = (255, 0, 185+ 100)   # 07 back_low
    MARKERS["09"] = (185, 0, 255+ 100)   # 08 left b_shoulder
    MARKERS["10"] = (0, 255, 185+ 100)   # 09 left f_shoulder
    MARKERS["11"] = (255, 185, 0+ 100)   # 10 left upperarm
    MARKERS["12"] = (132, 0, 255+ 100)   # 11 left forearm
    MARKERS["13"] = (0, 255, 132+ 100)   # 22 left hand
    MARKERS["14"] = (255, 132, 0+ 100)   # 12 right b_shoulder
    MARKERS["15"] = (224, 255, 0+ 100)   # 13 right f_shoulder
    MARKERS["16"] = (0, 225, 255+ 100)   # 14 right upperarm
    MARKERS["17"] = (255, 0, 225+ 100)   # 15 right forearm   
    MARKERS["18"] = (138, 255, 0+ 100)   # 24 right hand    
    MARKERS["19"] = (0, 138, 255+ 100)   # 16 left pelvis
    MARKERS["20"] = (255, 0, 138+ 100)   # 17 left thigh
    MARKERS["21"] = (222, 0, 255+ 100)   # 18 left calf
    MARKERS["22"] = (0, 255, 222+ 100)   # 23 left foot
    MARKERS["23"] = (255, 222, 0+ 100)   # 19 right pelvis
    MARKERS["24"] = (97, 0, 255+ 100)    # 20 right thigh
    MARKERS["25"] = (0, 255, 97+ 100)    # 21 right calf
    MARKERS["26"] = (255, 95, 0+ 100)    # 25 right foot


    return MARKERS

joint_selection = [
    0,  # Hips
    2,  # Spine1
    3,  # Spine2
    5,  # Neck
    8,  # Head
    10, # RightArm
    11, # RightForeArm
    12, # RightHand
    16, # LeftArm
    17, # LeftForeArm
    18, # LeftHand
    21, # RightUpLeg
    22, # RightLeg
    23, # RightFoot
    25, # RightToeBase
    # 25, # LeftUpLeg
    # 27, # LeftLeg
    # 28, # LeftFoot
    # 30, # LeftForeFoot
    # 31, # LeftToeBase
    27, # LeftUpLeg
    28, # LeftLeg
    29, # LeftFoot
    31, # LeftToeBase
]

# 0	Hips
# 1	Spine
# 2	Spine1
# 3	Spine2
# 4	Spine3
# 5	Neck
# 6	Neck1
# 7	Head
# 8	HeadEnd
# 9	RightShoulder
# 10	RightArm
# 11	RightForeArm
# 12	RightHand
# 13	RightHandThumb1
# 14	RightHandMiddle1
# 15	LeftShoulder
# 16	LeftArm
# 17	LeftForeArm
# 18	LeftHand
# 19	LeftHandThumb1
# 20	LeftHandMiddle1
# 21	RightUpLeg
# 22	RightLeg
# 23	RightFoot
# 24	RightForeFoot
# 25	RightToeBase
# 26	RightToeBaseEnd
# 27	LeftUpLeg
# 28	LeftLeg
# 29	LeftFoot
# 30	LeftForeFoot
# 31	LeftToeBase
# 32	LeftToeBaseEnd

joint_selection2 = [
    0,  # Hips
    11,  # Spine1
    12,  # Spine2
    20,  # Neck
    21,  # Head
    24, # RightArm
    25, # RightForeArm
    26, # RightHand
    14, # LeftArm
    15, # LeftForeArm
    16, # LeftHand
    6, # RightUpLeg
    7, # RightLeg
    8, # RightFoot
    9, # RightToeBase
    # 25, # LeftUpLeg
    # 27, # LeftLeg
    # 28, # LeftFoot
    # 30, # LeftForeFoot
    # 31, # LeftToeBase
    1, # LeftUpLeg
    2, # LeftLeg
    3, # LeftFoot
    4, # LeftToeBase
]

marker_mapping_sfu_2_h4d = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    
]

