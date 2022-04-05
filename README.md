# Agisoft camera location to NERF conversion tool
This tool is for use with https://github.com/NVlabs/instant-ngp and allows the use of Agisoft Metashape camera locations

## Installation
Copy agi2nerf.py file into instant-ngp\scripts folder

## Usage
Use Agisoft Metashape to align cameras
Export cameras
```
File -> Export -> Export Cameras...
```

Save example.xml as Agisoft XML

Create a new example folder in instant-ngp\data\nerf

Copy example.xml into instant-ngp\data\nerf\example
Copy all images used into instant-ngp\data\nerf\example\images

## Commands
Example:
```
instant-ngp\data\nerf\example$ python agi2nerf.py --xml_in ./example.xml
```

Arguments:

| Argument    | Default Value   | Description               |
|-------------|-----------------|---------------------------|
| --xml_in    | None            | specify xml file location |
| --out       | transforms.json | specify output file path  |
| --imgfolder | ./images/       | location of image folder  |
| --imgtype   | jpg             | ex.: jpg, png, ...        |