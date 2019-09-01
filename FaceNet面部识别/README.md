# Face Recognition based on Facenet
This is a face recognition program which is based on [Facenet](https://github.com/davidsandberg/facenet). 

In this project, the following functions can implemented.

## Prerequisite
- tensorflow==1.7
- scipy
- scikit-learn
- opencv-python
- h5py
- matplotlib
- Pillow
- requests
- psutil

## Training our own classifier
We can trian our own classifier by using the raw image dataset. The reference code:
	
	python train.py \
	>${raw_data_full_path} \
	>${training_data_full_path} \
	>${pre-trained_model_full_path}\
	>${classifier_full_path} 
	
	'raw_data_full_path': Path to the data directory containing the raw face images.
	'training_data_full_path': Path to the data directory containing the aligned raw face images.
	'pre-trained_model_full_path': Path to the pre-trained model
	'classifier_full_path': Path to the trained classifier
## Classify the new image
After we trained the classifier, we can use it to classify the new image uploaded by user. The reference code:
	
	python classify.py \
	${new_image_full_path} \
	${aligned_new_image_full_path} \
	${raw_data_full_path} \
	${pre-trained_model_full_path} \
	${classifier_full_path}
	
	'new_image_full_path': Path to the data directory containing the new image uploaded by user.
	'aligned_new_image_full_path': Path to the data directory containing the aligned new image.
	'raw_data_full_path': Path to the data directory containing the raw face images.
	'pre-trained_model_full_path': Path to the pre-trained model
	'classifier_full_path': Path to the trained classifier
	
## Copy the new image to our dataset
We can move the new image uploaded by user to our raw image dataset. The reference coed: 

	python get_image_path.py \
	${raw_data_full_path} \
	${new_image_full_path} \
	${user_id} \
	${label} \
	${image_name}
	
	'raw_data_full_path': Path to the data directory containing the raw face images.
	'new_image_full_path': Path to the data directory containing the new image uploaded by user.
	'user_id': The id of the user.
	'label': The label given by user.
	'image_name': The full name of the new image.

