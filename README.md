# EddyData

Keras implementation based on Tensorflow and dataset SCSE-Eddy for article "A Deep Framework for Eddy Detection and Tracking from Satellite Sea Surface Height Data"

### The Structure of Our FrameWork

![756846887685](EddyData/structureforreadme.jpg)

### Example for Detection Result

![1552712826999](G:\DeepLearning\Ocean_Eddy_Article\template_7\resultforreadme.png)



### Download the Dataset SCSE-Eddy

 Unzip and locate it  



### Train and Validate 

```
python3 FrameWork.py
```

store the detection results on your own path



### Example for Tracking Result

![7654679457](G:\DeepLearning\Ocean_Eddy_Article\template_7\trackforreadme.png)



### Tracking

load the detection results and then

```
python3 trace.py
```



#### Comparison of Number of Eddies Detected by Methods

| Methods | N-Cyclone | N-Anticyclone | N-All |
| :-----: | :-------: | :-----------: | :---: |
|   PET   |   4437    |     3737      | 8210  |
| EddyNet |   4502    |     3794      | 8296  |
| PSPNet  |   4614    |     3906      | 8520  |
|  Ours   |   4662    |     3952      | 8614  |



### Others

 Please refer to the citations in this paper for SCSE-Eddy dataset. 



##### We look forward to new breakthroughs for scholars in the future !
