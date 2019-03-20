# UpDrive_tools
## Install  
1. Suggest to create a new virtualenv for the package:  
`$ mkvirtualenv updrive-tools`  
`$ workon updrive-tools`  
`$ cd TO_YOUR_WORKSPACE`  
`$ git clone https://github.com/brade31919/UpDrive_tools`  

2. Install python dependencies  
`$ cd UpDrive_tools`  
`$ pip install -r requirements.txt`  

## Usage  
Provide command lines help function:  
`$ cd python`  
`$ python undistorter.py --h`  
Simple usage example:  
`$ python undistorter.py -inputs_dir INPUTS_FOLDER -outputs_dir OUTPUTS_FOLDER -yaml_file CALIBRATION_FILE`  
Set verbose and specific scale factor for cropping the center area (default scale is 5):  
`$ python undistorter.py --scale 20 --v -inputs_dir INPUTS_FOLDER -outputs_dir OUTPUTS_FOLDER -yaml_file CALIBRATION_FILE`  

  
Note:   
The 'inputs_dir' should point to the folder which has 4 sub-folders (front, left, rear, right)  
The 'outputs_dir' should point to the folder which 4 sub-folders do not exist!  

## Feature   
* Stop the process if the outputs sub-folders already existing (safe issue)  
* For different sub-folders' naming, please modify the list in undistorter.py  

## Troubleshooting  
Q. ImportError --> undefined symbol: PyCObject_Type  
A. Remember to Disable ROS environment setpu.bash   

## Developer  
* Hao-Chih Lin (hlin@ethz.ch)  
* Juan-Ting Lin (julin@ethz.ch)  

## License  
Apache License 2.0  
