import numpy as np
import cv2

# Check if plot libraries are installed
try:
	from pytransform3d.plot_utils import plot_box
	from pytransform3d.transform_manager import TransformManager

	import pytransform3d.camera as pc
	import pytransform3d.transformations as pytr

	import matplotlib.pyplot as plt
	from matplotlib.widgets import Slider, Button
	_plt = True
except ImportError as e:
	print("Plotting libraries not found. Skipping plotting.")
	_plt = False


###############################################################################
# START
# code taken from https://github.com/NVlabs/instant-ngp
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def central_point(out):
	# find a central point they are all looking at
	print("computing center of attention...")
	totw = 0.0
	totp = np.array([0.0, 0.0, 0.0])
	for f in out["frames"]:
		mf = np.array(f["transform_matrix"])[0:3,:]
		for g in out["frames"]:
			mg = g["transform_matrix"][0:3,:]
			p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
			if w > 0.01:
				totp += p*w
				totw += w

	totp /= totw
	print("The center of attention is: {}".format(totp)) # the cameras are looking at totp

	return totp

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	if image is None:
		print("Image not found:", imagePath)
		return 0
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	return fm

#END
###############################################################################


def reflect(axis, size=4):
	_diag = np.ones(size)
	_diag[axis] = -1
	refl = np.diag(_diag)
	return refl


def agiMat2Nerf(mat, _reflect=True):
	# return mat
	M = np.array(mat)

	# Swap axes
	M = M[[2,0,1,3],:]

	# if _reflect:
	# Reflect Z and Y axes
	M = ((M @ reflect(2)) @ reflect(1))

	return M


def draw_cameras(ax, out, camera_size):
	# Plot the camera positions
	for f in out['frames']:
		sensor_size = np.array([f["w"], f["h"]])

		intrinsic = np.eye(3)
		intrinsic[0,0] = f["fl_x"]
		intrinsic[1,1] = f["fl_y"]
		intrinsic[0,2] = f["cx"] if "cx" in f else sensor_size[0] / 2.0
		intrinsic[1,2] = f["cy"] if "cy" in f else sensor_size[1] / 2.0

		cam_mat = np.array(f["transform_matrix"])

		# Scale the camera position
		# cam_mat[0:3,3] *= scale

		# Reflect the camera back for plotting
		cam_mat = cam_mat @ reflect(1) @ reflect(2)

		pytr.plot_transform(ax, A2B=cam_mat, s=camera_size)
		pc.plot_camera(ax, cam2world=cam_mat, M=intrinsic,
						sensor_size=sensor_size,
						virtual_image_distance=camera_size)
		

def plot(out, origin, region, camera_size=0.1):

	# 3D plot the points and display them
	fig = plt.figure()
	
	ax = plt.axes(projection='3d')

	ax.set_xlabel('x')
	ax.set_ylabel('z')
	ax.set_zlabel('y')

	fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

	# Find the scene scale
	P = [np.array(f["transform_matrix"])[0:3,3] for f in out['frames']]
	pos_min = np.min(P, axis=0)
	pos_max = np.max(P, axis=0)
	# print("Scene size:", pos_max - pos_min)
	center = (pos_max + pos_min) / 2.0
	max_half_extent = max(pos_max - pos_min) / 2.0
	# print("Max half extent:", max_half_extent)

	# Plot the camera positions
	draw_cameras(ax, out, camera_size)

	# Plot the origin for reference
	pytr.plot_transform(ax, A2B=np.eye(4), s=1)

	if region is not None:
		# Plot the bounding box
		bbox_mat = region['transform_matrix']
		bbox_mat = bbox_mat @ reflect(1) @ reflect(2)
		bbox_mat[0:3,3] -= origin # Translate the bbox to match the center
		plot_box(ax, size=region['size'], A2B=bbox_mat, color='r', alpha=0.5)

	# Set the limits
	ax.set_xlim((center[0] - max_half_extent, center[0] + max_half_extent))
	ax.set_ylim((center[1] - max_half_extent, center[1] + max_half_extent))
	ax.set_zlim((center[2] - max_half_extent, center[2] + max_half_extent))

	# Create sliders to adjust scale of the scene
	# slider_scale = Slider(plt.axes([0.25, 0.05, 0.65, 0.03]), 'Scale', 0.01, 10.0, valinit=1.0)
	# plt.axes([0.25, 0.05, 0.65, 0.03])
	# def update(val):
	# 	scale = slider_scale.val
	# 	draw_cameras(ax, out, scale)
	# 	fig.canvas.draw_idle()

	# slider_scale.on_changed(update)

	plt.show()

	