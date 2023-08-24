# MSc_Dissertation
MSc Dissertation: Improving the detection of handwritten musical symbols with a synthesised dataset

README FILE

Project Objective: Improving detection of handwritten musical symbols with a synthesized dataset

Pranjali Hande
Student ID 220707639
Dr.George Fazekas
Big Data Science

This is a README file that provides steps to execute the dissertation source code. All these requirements need to be satisfied in order to execute the code successfully. This source code requires TensorFlow with GPU setup, hence, cannot be run directly (example: Google Colab). 
Due to size constraints, only some samples of the input dataset and generated images are uploaded as a supporting document. Below is the GitHub link which contains complete datasets and generated images for each model. 


**A.	Setup Requirements:**
To execute this source code there are some prerequisites needed to be performed on GPU-enabled machines. This code was developed on QMUL EECS lab for Student remote server LUNA with user EC22221

•	Create a new python virtual machine by referring to: https://www.tensorflow.org/install/pip

1.	System Requirement: Ubuntu 16.04 or higher (64-bit)
2.	To install TensorFlow with GPU setup, first need to create an environment using Miniconda. To install Miniconda use the following commands:
   
      1. curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
      2. bash Miniconda3-latest-Linux-x86_64.sh
      3. Use “conda -v” to verify.
  	
3.	Create Conda Environment:
       1. conda create --name <env_name> python=3.9
       2. conda activate <env_name>
	
4.  GPU Setup:
    Verify whether NVIDIA GPU Driver is installed using the below command: “nvidia-smi”
    Then install CUDA and cuDNN with conda and pip.
    
      1. conda install -c conda-forge cudatoolkit=11.8.0
      2. pip install nvidia-cudnn-cu11==8.6.0.163
      3. CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
      4. export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
      5. mkdir -p $CONDA_PREFIX/etc/conda/activate.d
      6. echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
      7. echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh


•	Install the below packages using “pip install” in the virtual environment.

      1.	tensorflow-gpu
      2.	nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
      3.	ipykernel
      4.	jupyterlab
      5.	notebook
      6.	git+https://github.com/tensorflow/examples.git
      7.	tensorflow_datasets
      8.	matplotlib
      9.	cv2
      10.	opencv-python
      11.	openssl
      12.	clean-fid
      13.	tensorflow-addons==0.16.1
      Verify the installed packages using the command “conda list <package_name>”


•	We are using 'pix2pix.unet_generator' from the tesnorflow_examples package. This code has a hardcoded input channel as 3 (for RGB images), but we have grayscale images as input which requires a number of input channels as 1.  Hence, in order to fix this, I have made changes in below file:

      - ‘/<path_where_conda_installed>/lib/python3.9/site-packages/tensorflow_examples/models/pix2pix/pix2pix.py’.
  
In my environment, this path is:

      - “/home/ec22221/miniconda3/envs/cycleGan_tf_conda/lib/python3.9/site-packages/tensorflow_examples/models/pix2pix/pix2pix.py”

We added input_channels as an argument to the method. Made a change in Discriminator as well. Below screenshots shows which methods are updated. In order to run this code, this change is mandatory. The full file is also uploaded in a supporting document to verify.

        1. Line No: 257, added parameter “**input_channels**”
           def unet_generator(output_channels, **input_channels**, norm_type='batchnorm'):
        2. Line No: 297, update “**input_channels**”
           inputs = tf.keras.layers.Input(shape=[None, None, **input_channels**])
        3. Line No: 318, added parameter “**input_channels**”
           def discriminator(**input_channels**, norm_type='batchnorm', target=True):
        4. Line No: 331, update “**input_channels**”
           inp = tf.keras.layers.Input(shape=[None, None, **input_channels**], name='input_image')
        5. Line No: 335, update “**input_channels**”
           tar = tf.keras.layers.Input(shape=[None, None, **input_channels**], name='target_image')
      


**B.	Dataset Pre-processing**

In this project, we are using 2 datasets, namely, DoReMi (Printed sheet Music) and CVC-MUSCIMA (Handwritten Sheet Music). Below are the respective websites from where we have downloaded them:

DoReMi: https://github.com/steinbergmedia/DoReMi/releases/tag/v1.0

CVC-MUSCIMA Database: http://pages.cvc.uab.es/cvcmuscima/index_database.html

We have in total 3 models to evaluate the performance. Dataset Requirements are different for them. Following are the perquisites needed to be performed in order to execute all models.

•	Dataset pre-processing requirement to run Complete_Images_UNet_model:

  This model uses the complete music sheet as exists in public datasets. Some pre-processing is required in order to run this model, which are mentioned below:

  1.	The Doremi Dataset has images with 3 input channels. We need to convert all these images to have one input channel to keep consistency across both datasets.
      Code available in “utils.ipynb” file to achieve this. Follow the markdown [Run Below cell convert all images of DoReMi dataset to single input channel] to locate the exact cell.
      
  2. The CVC-MUSCIMA Dataset has images from all musicians in different folders. In order to process them, they all need to collect in one single folder. 
     Code available in “utils.ipynb” file to achieve this. Follow the markdown [Run below cell to copy all subfloder of CVC-MUSCIMA to one folder] to locate the exact cell.

  3. Divide all the images into Train and Test folders in order to create TensorFlow datasets later in the model.

    Code available in “utils.ipynb” file to achieve this. Follow the markdown [Run below cell to create Train and Test folder for each dataset, DoReMi and CVC-MUSCIMA] to locate the exact cell.
    Note: All these pre-processed datasets are used to execute the model. A sample of these updated datasets is added as a part of supporting documentation. Also, the full dataset is present at the GitHub location: 

•	Dataset pre-processing requirement to run short images for UNet Model and ResNet Model:

    As the initial attempt of using complete sheet music, did not produce satisfactory results, we have used a data augmentation technique. For both datasets DoReMi and CVC-MUSCIMA, we cropped all images by their staves. 
    We have directly used these staves’ cropped datasets. Some of the samples are added in supporting documents and the full dataset available on GitHub: 

    Due to this, all images become rectangular in shape. We are using a pix2pix generator which only accepts square images, we again cropped all these images in square shape. Code to crop all these images in square shape is available in the file: ‘Generated_Square_Images.ipynb’
    After generating these square shape images, we need to follow the same preprocessing steps as mentioned for the last model.

    1. Update DoReMi staves dataset images to convert in single channel (Grayscale image)
    2. Get all images of CVC-MUSCIMA dataset in one single folder.
    3. Divide all the images into Train and Test folders in order to create TensorFlow datasets later in the model.

    Once all these steps are completed, make sure all the path locations are updated as per the environment used, and then we can run all the models and evaluate the generated images.


**C.	Model Evaluation:**

  The code for model evaluation is presented in file ‘Model_Evaluation_Using_FID_and_IS.ipynb’
  
  1.	FID evaluation: Please provide the path below two paths to calculate FID:
  First: The path of newly generated handwritten music sheet images from the model
  Second: The path of corresponding given input printed music sheet images from the model

  3.	IS evaluation: Please provide the path of the generated handwritten music sheet images to calculate IS.



