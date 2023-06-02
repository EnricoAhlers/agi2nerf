
import argparse
import xml.etree.ElementTree as ET
import math
import numpy as np
from copy import deepcopy as dc

from utils import agiMat2Nerf, sharpness, central_point, plot, _plt

import json

from tqdm import tqdm
from pathlib import Path


def parse_args():
	parser = argparse.ArgumentParser(description="convert Agisoft XML export to nerf format transforms.json")

	parser.add_argument("--xml_in", help="specify xml file location") #TODO: Chang to positional argument
	parser.add_argument("--out", dest="path", default="transforms.json", help="output path")
	parser.add_argument("--imgfolder", default="./images/", help="location of folder with images")
	parser.add_argument("--imgtype", default="jpg", help="type of images (ex. jpg, png, ...)")
	parser.add_argument("--aabb_scale", default=16, type=int, help="size of the aabb, default is 16")
	parser.add_argument("--plot", action="store_true", help="plot the cameras and the bounding region in 3D")
	parser.add_argument("--scale", default=1.0, type=float, help="scale the scene by a factor")
	parser.add_argument("--no_scale", action="store_true", help="DISABLES the scaling of the cameras to the bounding region")
	parser.add_argument("--no_center", action="store_true", help="DISABLES the centering of the cameras around the computed central point")
	parser.add_argument("--camera_size", default=0.1, type=float, help="size of the camera in the 3D plot. Does not affect the output.")
	
	parser.add_argument("--debug_ignore_images", action="store_true", help="IGNORES the images in the xml file. For debugging purposes only.")
	
	args = parser.parse_args()
	return args


def parse_region(xml_root):
	"""
	Parse the region xml
	The xml is formatted as follows:

	<region>
		<center>0 0 0.5</center>
		<size>1 1 1</size>
		<R>1 0 0 0 1 0 0 0 1</R>
    </region>
	"""
	region = xml_root.find('.//region')
	
	center = np.array([float(i) for i in region.find('center').text.split()])
	size = np.array([float(i) for i in region.find('size').text.split()])
	rotation = np.array([float(i) for i in region.find('R').text.split()]).reshape(3,3)

	mat = np.eye(4)
	mat[:3,:3] = rotation.T # Why transpose? Don't ask questions...
	mat[:3,3] = center
	
	return mat, size


def parse_components(xml_root):
	"""
	Parse the transform and region from components of the xml
	The xml is formatted as follows:
    <components next_id="1" active_id="0">
      <component id="0" label="Component 1">
		<transform>
		<rotation locked="true">1 0 0 0 1 0 0 0 1</rotation>
		<translation locked="true">0 0 0</translation>
		<scale locked="true">1</scale>
		</transform>
		<region>
			<center>0 0 0.5</center>
			<size>1 1 1</size>
			<R>1 0 0 0 1 0 0 0 1</R>
		</region>
      </component>
    </components>
	"""
	# http://wiki.agisoft.com/wiki/Coordinate_System_to_Bounding_Box.py
	# https://www.agisoft.com/forum/index.php?topic=6176.0

	comp = xml_root.find('.//components/component')

	if comp is None:
		return None
	
	xml_tform = comp.find('transform')
	xml_region = comp.find('region')

	if xml_tform is None:
		scene = (np.eye(4), 1.0)
	else:
		rotation = np.array([float(i) for i in xml_tform.find('rotation').text.split()]).reshape(3,3)
		translation = np.array([float(i) for i in xml_tform.find('translation').text.split()])
		scale = np.array([float(i) for i in xml_tform.find('scale').text.split()])

		mat = np.eye(4)
		mat[:3,:3] = rotation
		mat[:3,3] = translation
		# mat = pytr.scale_transform(mat, s_d=scale)

		scene = mat, scale

	if xml_region is None:
		region = (None, None)
	else:
		center = np.array([float(i) for i in xml_region.find('center').text.split()])
		size = np.array([float(i) for i in xml_region.find('size').text.split()])
		rotation = np.array([float(i) for i in xml_region.find('R').text.split()]).reshape(3,3)

		mat = np.eye(4)
		mat[:3,:3] = rotation.T # Why transpose? Don't ask questions...
		mat[:3,3] = center

		region = mat, size
	
	# print(scene, region)
	return scene, region


def parse_xform(cam):
	xform = cam.find('transform')

	# These are unused right now
	# rotation_covariance = cam.find('rotation_covariance')
	# location_covariance = cam.find('location_covariance')

	if xform is None:
		return None
	
	mat = np.array([float(i) for i in xform.text.split()]).reshape(4,4)

	return mat


def parse_camera(cam):

	if not len(cam):
		return None

	if(cam.find('transform') == None):
		return None
	
	# Get the camera label and sensor id
	# So we can match the sensor to the camera
	label = cam.get("label")
	sensor_id = cam.get("sensor_id")

	current_camera = dict()
	current_camera['label'] = str(label)
	current_camera['sensor_id'] = int(sensor_id)
	current_camera['transform_matrix'] = parse_xform(cam)
	
	return current_camera


def parse_sensor(sensor):
	out = dict()

	# Get the sensor id
	id = sensor.get("id")
	out['id'] = int(id)

	# Calibration coefficients and parameters
	# https://www.agisoft.com/pdf/metashape-pro_1_5_en.pdf
	# (F, Cx, Cy, B1, B2, K1, K2, K3, K4, P1, P2, P3, P4)

	calib = sensor.find('calibration')

	if (calib is None):
		print("No calibration found for sensor {}".format(id))
		
		# Get the sensor resolution
		res = sensor.find("resolution")
		w = float(res.get('width'))
		h = float(res.get('height'))

		"""	
		Get the pixel width and height
		The xml is formatted as follows:
		<sensor id="" label="" type="frame">
			<resolution width="4000" height="6000"/>
			<property name="pixel_width" value=""/>
			<property name="pixel_height" value="0.0039083244579612101"/>
			<property name="focal_length" value="55"/>
			<property name="layer_index" value="0"/>
		"""

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

		out["camera_angle_x"] = camera_angle_x
		out["camera_angle_y"] = camera_angle_y
		out["fl_x"] = fl_pxl
		out["fl_y"] = fl_pxl
		out["w"] = w
		out["h"] = h
	else:
		res = calib.find("resolution")
		w = float(res.get('width'))
		h = float(res.get('height'))

		fl_x = float(calib.find('f').text)
		fl_y = fl_x
		
		k1 = float(calib.find('k1').text if calib.find('k1') is not None else -1)
		k2 = float(calib.find('k2').text if calib.find('k2') is not None else -1)
		p1 = float(calib.find('p1').text if calib.find('p1') is not None else -1)
		p2 = float(calib.find('p2').text if calib.find('p2') is not None else -1)
		cx = float(calib.find('cx').text if calib.find('cx') is not None else 0) + w/2
		cy = float(calib.find('cy').text if calib.find('cy') is not None else 0) + h/2

		camera_angle_x = math.atan(float(w) / (float(fl_x) * 2)) * 2
		camera_angle_y = math.atan(float(h) / (float(fl_y) * 2)) * 2
		
		out["camera_angle_x"] = camera_angle_x
		out["camera_angle_y"] = camera_angle_y
		out["fl_x"] = fl_x
		out["fl_y"] = fl_y
		out["k1"] = k1
		out["k2"] = k2
		out["p1"] = p1
		out["p2"] = p2
		out["cx"] = cx
		out["cy"] = cy
		out["w"] = w
		out["h"] = h
		
	return out
	

def calibration(root, stems, _scale=1.0, _no_scale=False, _ignore_images=False):
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

	sensors = [parse_sensor(s) for s in sensors]
	cameras = [parse_camera(c) for c in cameras]

	# Remove empty cameras
	cameras = [c for c in cameras if c]

	# Transform the cameras to the component's coordinate system
	scene, region = parse_components(root)

	scene_mat, scene_scale = scene
	region_mat, region_scale = region
	
	scene_scale = scene_scale if not _no_scale else 1.0
	scene_scale *= _scale

	for c in cameras:
		M = c['transform_matrix']

		if (scene_mat is not None) & (scene_scale is not None):
			M[:3,3] = M[:3,3] * scene_scale
			M = np.dot(scene_mat, M)
		
		c['transform_matrix'] = agiMat2Nerf(M)

	calib = []

	# Match sensors to cameras
	#TODO: There's probably a better way to do this...
	for c in cameras:
		for s in sensors:
			if(c['sensor_id'] == s['id']):
				calib.append((dc(c), dc(s)))
				break
	
	print("\nFound {} cameras and {} sensors".format(len(cameras), len(sensors)))
	print("\nFound {} matching cameras and sensors".format(len(calib)))

	frames = []

	pbar = tqdm(total=len(root[0][2]))

	for camera, sensor in calib:
		pbar.update(1)

		if (camera is None) or (sensor is None):
			print('No camera or sensor found')
			continue

		if not _ignore_images:
			# Check if label is in image folder
			label = [str(f) for f in stems if(str(f) in camera['label'])]

			if(len(label) == 0):
				print('No matching image found for: {}'.format(camera['label']))
				continue

			imagePath = IMGFOLDER + '/' + label[0] + "." + IMGTYPE

			# Check if image exists
			if(Path(imagePath).is_file() == False):
				print('Image not found in path: {}'.format(imagePath))
				continue

			# Set the image path
			camera["file_path"] = imagePath
			camera["sharpness"] = sharpness(imagePath)

		del camera['label']
		del camera['sensor_id']
		del sensor['id']

		frame = sensor
		frame.update(camera)
		frames.append(frame)
	
	if (region_mat is None) or (region_scale is None):
		print('No bounding region found')
		region = None
	else:
		if (scene_mat is not None) & (scene_scale is not None):
			region_mat[:3,3] *= scene_scale # Scale the transform
			region_mat = np.dot(scene_mat, region_mat) # Rotate the bbox to match the scene
			region_scale *= scene_scale

		region_mat = agiMat2Nerf(region_mat) # Convert to the coordinates
		
		region = dict(transform_matrix=region_mat, size=region_scale)
	
	return frames, region


if __name__ == "__main__":
	args = parse_args()
	XML_LOCATION = args.xml_in
	IMGTYPE = args.imgtype
	IMGFOLDER = args.imgfolder

	files = list(Path(IMGFOLDER).glob('*.{}'.format(IMGTYPE)))
	stems = list([f.stem for f in files])

	# Check if the files path has images in it
	if(len(files)==0) & (args.debug_ignore_images==False):
		print('No images found in folder: {}'.format(IMGFOLDER))
		exit()

	out = dict()

	out['aabb_scale'] = args.aabb_scale

	with open(XML_LOCATION, "r") as f:
		xml_root = ET.parse(f).getroot()
		
		# See issue for multi camera support
		# https://github.com/NVlabs/instant-ngp/discussions/797

		frames, region = calibration(xml_root, stems,
			       			args.scale, args.no_scale, args.debug_ignore_images)

	out['frames'] = frames

	if args.no_center:
		center = np.zeros(3)
	else:
		# Compute the center of attention
		center = central_point(out)

	# Set the offset and convert to list
	for f in out["frames"]:
		f["transform_matrix"][0:3,3] -= center
		f["transform_matrix"] = f["transform_matrix"].tolist()
	
	with open(args.path, "w") as f:
		json.dump(out, f, indent=4)
	
	if _plt & args.plot:
		plot(out, center, region, args.camera_size)