import torch
import numpy as np
from  src.data.datasets.human_pose.H4DIR.importers.enums import (
    joint_selection
)
from  src.data.datasets.human_pose.H4DIR.importers.markermap import (
    S1S4_Mapping, 
    S2S3_Mapping
)

__all__ = [
    'load_3d_data',
]

def load_3d_data(filename, filename_rs, markers_out, joints_out, \
        rs=True, random_rot=None, augment=False, S1S4=True, \
        scale_on_image=1.25, data_type=torch.float32):
    # ground truth data
    file = open(filename, "r")
    lines = file.readlines()
    raw_points = np.zeros([markers_out + joints_out, 3], dtype=float)
    # num in the original ground truth data
    NUM_OF_MARKERS_IN_GT = 53
    NUM_OF_JOINTS_IN_GT = 33
    NUM_OF_LINES = NUM_OF_MARKERS_IN_GT + NUM_OF_JOINTS_IN_GT
    line_counter = 0
    j_counter = 0
    assert(len(lines) == NUM_OF_LINES)
    for line in lines:
        index = line_counter % NUM_OF_LINES
        if (joints_out == 0 and index == markers_out):
            break
        values = line.split(' ')     
        x_gt = float(values[2])
        y_gt = float(values[3])
        z_gt = float(values[1])
        if (index < NUM_OF_MARKERS_IN_GT and index < markers_out):
            raw_points[j_counter, 0] = x_gt
            raw_points[j_counter, 1] = -y_gt # (-) for imaging
            raw_points[j_counter, 2] = z_gt
            j_counter += 1   
        elif (index - NUM_OF_MARKERS_IN_GT) < NUM_OF_JOINTS_IN_GT and (index - NUM_OF_MARKERS_IN_GT) in joint_selection:            
            raw_points[j_counter, 0] = x_gt
            raw_points[j_counter, 1] = -y_gt # (-) for imaging
            raw_points[j_counter, 2] = z_gt
            j_counter += 1   
        line_counter += 1
    # raw points list to be filled either with raw or noisy vicon data
    raw_rs_points_list = []
    if (rs):
        file_rs = open(filename_rs, "r")
        lines_rs = file_rs.readlines()
        for line in lines_rs:
            values = line.split(' ') 
            x_rs = float(values[1])
            y_rs = float(values[2])
            z_rs = float(values[0])
            # this is thresholding the floor and the top. TODO: to be better investigated
            if y_rs > -830 and y_rs < 1170:
                raw_rs_points_list.append([x_rs, -y_rs, z_rs]) # (-) for imaging
        rs_raw_points = np.asarray(raw_rs_points_list)        
    else:
        raw_rs_points_list = raw_points
    # subtraction of CoM and rotational augmentation
    com = np.mean(rs_raw_points, axis=0)
    raw_points -= com
    rs_raw_points -= com
    if augment:
        raw_points = random_rot.apply(raw_points)
        rs_raw_points = random_rot.apply(rs_raw_points)

    points = np.zeros([(markers_out + joints_out), 3], dtype=float)
    rs_points = rs_raw_points   

    minval_z = 100000
    maxval_z = 0
    minval_y = minval_z
    maxval_y = maxval_z
    minval_x = minval_z
    maxval_x = maxval_z
    counter = 0    

    for index in range(raw_points.shape[0]): 
        if (index < markers_out):
            if (S1S4):
                points[index] = raw_points[S1S4_Mapping[index]]
            else:
                points[index] = raw_points[S2S3_Mapping[index]]
        else:
            points[index] = raw_points[index]
        counter += 1
            
    minval_x = np.minimum(np.min(rs_points[..., 0]), np.min(points[..., 0]))
    maxval_x = np.maximum(np.max(rs_points[..., 0]), np.max(points[..., 0]))
    minval_y = np.minimum(np.min(rs_points[..., 1]), np.min(points[..., 1]))
    maxval_y = np.maximum(np.max(rs_points[..., 1]), np.max(points[..., 1]))
    minval_z = np.minimum(np.min(rs_points[..., 2]), np.min(points[..., 2]))
    maxval_z = np.maximum(np.max(rs_points[..., 2]), np.max(points[..., 2]))
    
    tcom = torch.from_numpy(com).type(torch.FloatTensor)
    scale = torch.zeros([3], dtype=torch.float32)
    gt_markers = torch.zeros([markers_out, 3])                  
    gt_joints = torch.zeros([joints_out, 3])
    scale_on_image_offset = (1.0 - 1.0 / scale_on_image) / 2.0
    for i in range(0, markers_out + joints_out):
        scale[0] = scale_on_image * float(maxval_x - minval_x)
        scale[1] = scale_on_image * float(maxval_y - minval_y)
        scale[2] = scale_on_image * float(maxval_z - minval_z)

        norm_depth_value = (float(points[i][2]) - float(minval_z)) \
            / scale[2] \
        + scale_on_image_offset
        norm_x_value = (float(points[i][0]) - float(minval_x)) \
                / scale[0] \
        + scale_on_image_offset
        norm_y_value = (float(points[i][1]) - float(minval_y)) \
                / scale[1] \
        + scale_on_image_offset

        if (i < markers_out):
            gt_markers[i][0] = norm_x_value
            gt_markers[i][1] = norm_y_value
            gt_markers[i][2] = norm_depth_value
        else:
            gt_joints[i-markers_out][0] = norm_x_value
            gt_joints[i-markers_out][1] = norm_y_value
            gt_joints[i-markers_out][2] = norm_depth_value            

    rs_markers_ori = torch.zeros([rs_raw_points.shape[0], rs_raw_points.shape[1]]) 

    tcom[0] += - scale[0] / 2 + scale[0] / scale_on_image / 2 + float(minval_x)
    tcom[1] += - scale[1] / 2 + scale[1] / scale_on_image / 2 + float(minval_y)
    tcom[2] += - scale[2] / 2 + scale[2] / scale_on_image / 2 + float(minval_z)    

    for i in range(rs_raw_points.shape[0]):
        norm_depth_value = (float(rs_points[i][2]) - float(minval_z)) \
            / scale[2] \
        + scale_on_image_offset
        norm_x_value = (float(rs_points[i][0]) - float(minval_x)) \
                / scale[0] \
        + scale_on_image_offset
        norm_y_value = (float(rs_points[i][1]) - float(minval_y)) \
                / scale[1] \
        + scale_on_image_offset

        rs_markers_ori[i][0] = norm_x_value
        rs_markers_ori[i][1] = norm_y_value
        rs_markers_ori[i][2] = norm_depth_value

    return rs_markers_ori, gt_markers, gt_joints, scale, tcom
    
   

