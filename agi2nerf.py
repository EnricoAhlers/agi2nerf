
import argparse
import xml.etree.ElementTree as ET
import math
import cv2
import numpy as np

import json

from tqdm import tqdm
from pathlib import Path

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
	print(totp) # the cameras are looking at totp
	for f in out["frames"]:
		f["transform_matrix"][0:3,3] -= totp
		f["transform_matrix"] = f["transform_matrix"].tolist()
	return out

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

# Copyright (C) 2022, Enrico Philip Ahlers. All rights reserved.


def parse_args():
	parser = argparse.ArgumentParser(description="convert Agisoft XML export to nerf format transforms.json")

	parser.add_argument("--xml_in", default="", help="specify xml file location")
	parser.add_argument("--out", default="transforms.json", help="output path")
	parser.add_argument("--imgfolder", default="./images/", help="location of folder with images")
	parser.add_argument("--imgtype", default="jpg", help="type of images (ex. jpg, png, ...)")
	args = parser.parse_args()
	return args

def reflectZ():
	return [[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]]

def reflectY():
	return [[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]

def matrixMultiply(mat1, mat2):
	return np.array([[sum(a*b for a,b in zip(row, col)) for col in zip(*mat2)] for row in mat1])

#END
###############################################################################

# Copyright (C) 2023, Josiah David Reeves. All rights reserved.

def parse_camera(cam):

	if not len(cam):
		return None

	if(cam.find('transform') == None):
		return None
	
	# Get the camera label
	id = cam.get("label").split('_')[3]

	current_camera = dict()
	current_camera.update({"id":id})
	matrix_elements = [float(i) for i in cam[0].text.split()]
	transform_matrix = np.array([[matrix_elements[0], matrix_elements[1], matrix_elements[2], matrix_elements[3]], [matrix_elements[4], matrix_elements[5], matrix_elements[6], matrix_elements[7]], [matrix_elements[8], matrix_elements[9], matrix_elements[10], matrix_elements[11]], [matrix_elements[12], matrix_elements[13], matrix_elements[14], matrix_elements[15]]])
	
	#swap axes
	transform_matrix = transform_matrix[[2,0,1,3],:]

	#reflect z and Y axes
	current_camera.update({"transform_matrix":matrixMultiply(matrixMultiply(transform_matrix, reflectZ()), reflectY())} )

	return current_camera
	
def parse_sensor(sensor):
	# Calibration coefficients and parameters
	# https://www.agisoft.com/pdf/metashape-pro_1_5_en.pdf
	# (F, Cx, Cy, B1, B2, K1, K2, K3, K4, P1, P2, P3, P4)

	calib = sensor.find('calibration')

	out = dict()

	if (calib == None):
		# Calculate the focal if not provided

		# Get the sensor resolution
		res = sensor.find("resolution")
		w = float(res.get('width'))
		h = float(res.get('height'))

		# Get the pixel width and height
		# The xml is formatted as follows:
		# <sensor id="" label="" type="frame">
		# 	<resolution width="4000" height="6000"/>
		# 	<property name="pixel_width" value=""/>
		# 	<property name="pixel_height" value="0.0039083244579612101"/>
		# 	<property name="focal_length" value="55"/>
		# 	<property name="layer_index" value="0"/>

		properties = sensor.findall('property')
		pixel_width = float(properties[0].get('value'))
		pixel_height = float(properties[1].get('value'))

		# Get the focal length in mm
		focal_length = float(properties[2].get('value'))

		# Given the w, h, pixel_width, pixel_height, and focal_length
		# Calculate the focal length in pixels
		fl_pxl = (w * focal_length) / (w * pixel_width)

		camera_angle_x = math.atan(float(w) / (float(fl_pxl) * 2)) * 2
		camera_angle_y = math.atan(float(h) / (float(fl_pxl) * 2)) * 2

		out.update({"camera_angle_x": camera_angle_x})
		out.update({"camera_angle_y": camera_angle_y})
		out.update({"fl_x": fl_pxl})
		out.update({"fl_y": fl_pxl})
		# out.update({"k1": 0})
		# out.update({"k2": 0})
		# out.update({"p1": 0})
		# out.update({"p2": 0})
		# out.update({"cx": 0})
		# out.update({"cy": 0})
		out.update({"w": w})
		out.update({"h": h})
	else:
		res = calib.find("resolution")
		w = float(res.get('width'))
		h = float(res.get('height'))

		fl_x = float(calib.find('f').text)
		fl_y = fl_x
		k1 = float(calib.find('k1').text if calib.find('k1') else 0)
		k2 = float(calib.find('k2').text if calib.find('k2') else 0)
		p1 = float(calib.find('p1').text if calib.find('p1') else 0)
		p2 = float(calib.find('p2').text if calib.find('p2') else 0)
		cx = float(calib.find('cx').text if calib.find('cx') else 0) + w/2
		cy = float(calib.find('cy').text if calib.find('cy') else 0) + h/2

		camera_angle_x = math.atan(float(w) / (float(fl_x) * 2)) * 2
		camera_angle_y = math.atan(float(h) / (float(fl_y) * 2)) * 2
		
		out.update({"camera_angle_x": camera_angle_x})
		out.update({"camera_angle_y": camera_angle_y})
		out.update({"fl_x": fl_x})
		out.update({"fl_y": fl_y})
		out.update({"k1": k1})
		out.update({"k2": k2})
		out.update({"p1": p1})
		out.update({"p2": p2})
		out.update({"cx": cx})
		out.update({"cy": cy})
		out.update({"w": w})
		out.update({"h": h})
		
	return out
	
def calibration(root):
	'''
	Take the xml file and generate the calibration data

	The xml is formated as follows:
	<document>
		<chunk>
			<sensors>
				<sensor>
					<calibration>
						<resolution>
						<f>
						...
			<components>
					<component>
			<cameras>
					<camera>
						<transform>
						<rotation_covariance>
						<location_covariance>
			<reference>
			<region>
			<settings>
			<meta>
	'''

	sensors = root.findall('.//sensor')
	cameras = root.findall('.//camera')
	
	for (camera, sensor) in zip(cameras, sensors):
		yield parse_camera(camera), parse_sensor(sensor)


if __name__ == "__main__":
	args = parse_args()
	XML_LOCATION = args.xml_in
	IMGTYPE = args.imgtype
	IMGFOLDER = args.imgfolder

	files = Path(IMGFOLDER).glob('*.{}'.format(IMGTYPE))
	stems = list([f.stem for f in files])
	# print(list([f.stem for f in files]))

	out = dict()

	aabb_scale = 16

	out.update({"aabb_scale": aabb_scale})

	with open(XML_LOCATION, "r") as f:
		root = ET.parse(f).getroot()
		
		# See issue for multi camera support
		# https://github.com/NVlabs/instant-ngp/discussions/797

		pbar = tqdm(total=len(root[0][2]))

		frames = []

		for camera, sensor in calibration(root):
			pbar.update(1)

			if (camera is None) or (sensor is None):
				print('no camera or sensor found for id: {}'.format(camera['id']))
				continue

			# Check if label is in image folder
			label = [str(f) for f in stems if(camera['id'] in str(f))]

			if(len(label) == 0):
				print('no matching label found for id: {}'.format(camera['id']))
				continue

			del camera['id']

			imagePath = IMGFOLDER + '/' + label[0] + "." + IMGTYPE

			# Check if image exists
			if(Path(imagePath).is_file() == False):
				print('Image not found in path: {}'.format(imagePath))
				continue

			# Set the image path
			camera["file_path"] = imagePath
			camera["sharpness"] = sharpness(imagePath)

			frame = sensor
			frame.update(camera)
			frames.append(frame)
		
		out.update({"frames": frames})
		

	# 	frames = list()
	# 	pbar = tqdm(total=len(root[0][2]))
	    
	# 	for frame in root[0][2]:

	# 		if not len(frame):
	# 			continue

	# 		if(frame[0].tag != "transform"):
	# 			continue
			
	# 		id = frame.get("label").split('_')[3]
	# 		label = [str(f) for f in stems if(id in str(f))]
			
	# 		if(len(label) == 0):
	# 			print('no image found for id: {}'.format(id))
	# 			continue

	# 		imagePath = IMGFOLDER + '/' + label[0] + "." + IMGTYPE
	# 		# if(Path(imagePath).is_file() == False):
	# 		# 	continue
	# 		# print(imagePath)
	# 		# print("!!!!!!!!!!!")

	# 		current_frame = dict()
	# 		current_frame.update({"file_path": imagePath})
	# 		current_frame.update({"sharpness":sharpness(imagePath)})
	# 		matrix_elements = [float(i) for i in frame[0].text.split()]
	# 		transform_matrix = np.array([[matrix_elements[0], matrix_elements[1], matrix_elements[2], matrix_elements[3]], [matrix_elements[4], matrix_elements[5], matrix_elements[6], matrix_elements[7]], [matrix_elements[8], matrix_elements[9], matrix_elements[10], matrix_elements[11]], [matrix_elements[12], matrix_elements[13], matrix_elements[14], matrix_elements[15]]])
			
	# 		#swap axes
	# 		transform_matrix = transform_matrix[[2,0,1,3],:]

	# 		#reflect z and Y axes
	# 		current_frame.update({"transform_matrix":matrixMultiply(matrixMultiply(transform_matrix, reflectZ()), reflectY())} )
			
	# 		frames.append(current_frame)
	# 		pbar.update(1)
	# 	out.update({"frames": frames})

	out = central_point(out)

	with open("transforms.json", "w") as f:
		json.dump(out, f, indent=4)