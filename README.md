# Grab Computer VIsion 

required folders
- data
	- car_devkit
		#place all .mat files here
		- cars_annos.mat
		- cars_meta.mat
		- cars_test_annos.mat
		- cars_train_annos.mat
		- eval_train.mat
	- car_ims 
		#directly unzip from standford car dataset site
		#contains 16185 images

#ensure python 3.7 and tensorflow GPU environment is set up 
#run script using command

python .\base_design.py -f data

