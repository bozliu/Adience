# AI6121 Course project
Gender and Age Detection


## Dependencies

* Anaconda3 (Python 3.7.6, with Numpy etc.)
* PyTorch 1.6
* tensorboard, tensorboardX

## Dataset

[Adience Benchmark Gender And Age Classification](https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification/notebooks) dataset 

## Data Pre-processing 
`data.py`
Since some of images labels are None or unknow and some of age labels are a specific value instead of interval we wanted, we filter out these incomplete data and remains 16k images in data preprocessing.   

## File Structure

├── data.py  
└── main.py   
└── models.py  
└── mtcnn_process.py  
└── utils.py  


* `data.py` : data augumentation, data preprocessing 
* `main.py` is the main function of the project
* models is the file with different model structure. `models.py` is the one used in this project.
* utils are the utility function required by the project

## Run Scripts Example

`! python main1.py \  
--model=resnet18 \  
--pretrained \  
--scheduler=step \  
--optimizer=Adam \  
--gender-weight=0.5 \  
--drop-out=0.3\`  


