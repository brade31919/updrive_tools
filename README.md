# UpDrive_tools
## Usage
Provide command lines help function:  
`$ python undistorter.py --h`  
Simple usage example:  
`$ python undistorter.py -inputs_dir INPUTS_FOLDER -outputs_dir OUTPUTS_FOLDER -yaml_file CALIBRATION_FILE`  
  
Note:   
The 'inputs_dir' should point to the folder which has 4 sub-folders (front, left, rear, right)  
The 'outputs_dir' should point to the folder which 4 sub-folders do not exist!  

## Feature   
* Stop the process if the outputs sub-folders already existing (safe issue)  
* For different sub-folders' naming, please modify the list in undistorter.py  

