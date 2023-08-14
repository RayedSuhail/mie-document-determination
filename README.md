# Medical Informatics Engineering - Summer Intern Project

#### Project worked on by **Rayed Suhail Ahmad**
#### Start of Project **June 19, 2023**

## Introduction
This project aims to develop a deep learning network that utilizes a few-shot or one-shot approach to training. This network should be capable of determining similarities between two image documents of medical records so that new documents can be added and categorized accordingly. The use of few-shot or one-shot learning techniques helps us design the neural network without the need for extensive retraining.

## Objectives
- Development of a Deep Learning neural network capable of determining the document type based on a few-shot or one-shot learning approach
- Integrate neural network with an existing system with speed and accuracy
- Render a CodeBullet-themed video monitoring the process of development from start to finish

## Methodology
- Gather and preprocess available medical documents for use by model
- Develop and test different model architectures utilized for object classification by utilizing ML frameworks and libraries (Keras, TensorFlow, Pandas, NumPy)
- Train and validate a model using a subset of the document types and then confirm performance for the remaining documents
- [OPTIONAL] Test the performance of the model in classifying a subset of documents that may differ from the ones trained on
- Utilize different performance metrics such as accuracy, precision, and F-score for evaluation of model
- Record and maintain the workflow to be utilized for the final video creation
- Edit together and record voice-overs for explaining the project

## To-Do List of Tasks
- [x] ~Use ADP timecard for hour tracking~
- [x] ~Create Internship Charter Document~
- [x] ~Finish YT Playlist to gain information about AI~ ([Link](https://www.youtube.com/playlist?list=PLuIoQNgtU5vq1tJWHngx91N5ndkHpT2Mg))
- [ ] Gather dataset from Doug
- [ ] Setup access to Tesla hardware for model training
- [x] ~Gather alternate dataset for developing proof-of-concept~
- [x] ~Setup environment for model training~
- [ ] Prepare dataset to be used for training, testing, and validation purposes
- [ ] Design model architecture for document classification
- [ ] Choose appropriate hyperparameters to maximize performance
- [ ] Code appropriate algorithm for few-shot learning
- [ ] Record performance and visuals to be utilized in video

## Setting up Anaconda Environment
To utilize the code written in this repo, please follow the following instructions for installing Anaconda and creating a conda environment for running the code by utilizing the provided `environment.yml` file. To start, you need to first use this [link](https://www.anaconda.com/download) to download the Anaconda installer. Follow the steps for your system provided in the [installation guide](https://docs.anaconda.com/free/anaconda/install/index.html) to set up Anaconda with a graphical interface. Once you have followed all the steps and successfully installed Anaconda in your system, open the Anaconda Prompt console. Using the console, navigate to where you cloned this repository and type the following command to create a duplicate virtual environment to the one that was used to train the model:

`conda env create -f environment.yml`

Once the process completes, you can activate the environment by running the command:

`conda activate mieml`

While in this environment you can utilize the same packages that were utilized in creating the model by using the `python` command and running a Python console directly in the Anaconda Prompt. However, for using IDE we suggest `Spyder` as it allows for amazing graphical features like Variable Explorer, Python Terminal Manager, and Code Editor. Spyder can be started using the current conda environment by running the following command:

`spyder`

## Using this repo
Once you have your IDE and environment set up, you can easily run the files found in this repository. In case you are unable to see any output or face an error when running a piece of code, please ensure that you read the comments associated with the code block and understand the requirements of the function or snippet of code.