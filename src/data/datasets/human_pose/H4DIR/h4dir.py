import os
import random
from scipy.spatial.transform import Rotation as R
import logging
from src.data.datasets.human_pose.H4DIR.importers import (
	load_3d_data
)
from src.data.datasets.human_pose.H4DIR.importers import (
    get_depth_image_from_points,
	f_rotate_back,
    f_rotate_left,
    f_rotate_right
)

from torch.utils.data.dataset import Dataset

log = logging.getLogger(__name__)

__all__ = ["H4DIR"]

class H4DIR(Dataset):
	def __init__(self,
		root_path, 
		markers_out, 
		joints_out, 
		resolution, 
		views,
		augment, 
		rs,
		scale_res=5.0,
		x_range = 20.0,
		y_range = 360.0,
		z_range = 20.0
	):
		super(H4DIR,self).__init__()
		self.root_path = root_path
		self.markers_out = markers_out
		self.joints_out = joints_out
		self.resolution = resolution
		self.views = views
		self.augment = augment
		self.rs = rs
		self.scale_res = scale_res
		self.x_range = x_range
		self.y_range = y_range
		self.z_range = z_range

		if not os.path.exists(root_path):
			raise ValueError("{} does not exist, exiting.".format(root_path))

		self.data = {}
		# Iterate over each recorded folder
		for recording in os.listdir(root_path):
			data_path = os.path.join(root_path, recording)
			if not os.path.isdir(data_path):
				continue
			for file in os.listdir(data_path):
				full_filename = os.path.join(data_path, file)
				filename, ext = os.path.splitext(full_filename)
				if ext != ".txt" or "_rs" in filename: # TODO: refactor the data loading and labeling
					continue					
				splits = file.split("_")
				if len(splits) == 3 or len(splits) == 4:
					_type = splits[0]
					_id = splits[1]
					_view = splits[2].split('.')[0]
				else:
					continue
				unique_name = recording + "-" + str(_id)
				if _view not in self.views:
					continue
				if unique_name not in self.data:
					self.data[unique_name] = {}
				if _view not in self.data[unique_name]:
					self.data[unique_name][_view] = {}
				self.data[unique_name][_view][_type] = full_filename

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		key = list(self.data.keys())[idx]
		datum = self.data[key]
		datum_out = {}
		random_rot = R.from_euler('xyz', [random.random() * self.x_range - self.x_range // 2, random.random() * self.y_range - self.y_range // 2, random.random() * self.z_range - self.z_range // 2], degrees=True)

		if "3d" in self.views:
			rs_markers_f, gt_markers_f, gt_joints_f, scale, com = load_3d_data(
				datum["3d"]["txt"], 
				datum["3d"]["txt"].replace('gt', 'rs'),
				self.markers_out, 
				self.joints_out,
				rs=self.rs, 
				S1S4=True if "_S1_" in  datum["3d"]["txt"] or "_S4_" in datum["3d"]["txt"] else False, # TODO: to discuss the marker placement type
				random_rot=random_rot,
				augment=self.augment
			)

		for view in self.views:
			if (view == "3d"):
				continue
			else:
				if (view == 'f'):
					gt_markers = gt_markers_f.clone()
					gt_joints = gt_joints_f.clone()
					rs_markers = rs_markers_f.clone()
					depth_img = get_depth_image_from_points(self.resolution, self.scale_res, rs_markers.clone())
				elif (view == 'b'):
					gt_markers = f_rotate_back(gt_markers_f.clone())
					gt_joints = f_rotate_back(gt_joints_f.clone())        
					rs_markers = f_rotate_back(rs_markers_f.clone())
					depth_img = get_depth_image_from_points(self.resolution, self.scale_res, rs_markers)
				elif (view == 'l'):     
					gt_markers = f_rotate_left(gt_markers_f.clone())
					gt_joints = f_rotate_left(gt_joints_f.clone())        
					rs_markers = f_rotate_left(rs_markers_f.clone())
					depth_img = get_depth_image_from_points(self.resolution, self.scale_res, rs_markers)
				elif (view == 'r'):     
					gt_markers = f_rotate_right(gt_markers_f.clone())
					gt_joints = f_rotate_right(gt_joints_f.clone())        
					rs_markers = f_rotate_right(rs_markers_f.clone())
					depth_img = get_depth_image_from_points(self.resolution, self.scale_res, rs_markers)
				else:
					raise ("Error. View {} is not supported.", view)

			datum_out.update({
				view + "_depth" : depth_img.squeeze(0),
				view + "_gt_markers_3d" : gt_markers,
				view + "_gt_markers_2d" : gt_markers[..., :2],
				view + "_gt_joints_3d" : gt_joints,
				view + "_gt_joints_2d" : gt_joints[..., :2],
				view + "_scale" : scale,
				view + "_com" : com.unsqueeze(0)
				})
		
		return datum_out

	def get_data(self):
		return self.data