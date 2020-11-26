# EddyData

Keras implementation based on Tensorflow and dataset SCSE-Eddy for article "A Deep Framework for Eddy Detection and Tracking from Satellite Sea Surface Height Data"

### The Structure of Our FrameWork

![structure.jpg](https://github.com/zmokokokok/EddyData/blob/master/example/structure.jpg)

### Example for Detection Result

![result.jpg](https://github.com/zmokokokok/EddyData/blob/master/example/result.jpg)



### Download the Dataset SCSE-Eddy

 Unzip and locate it  



### Train and Validate 

```
python3 FrameWork.py
```

store the detection results on your own path



### Example for Tracking Result

![track.jpg](https://github.com/zmokokokok/EddyData/blob/master/example/track.jpg)



### Tracking

load the detection results and then

```
python3 trace.py
```



#### Comparison of Number of Eddies Detected by Different Methods

| Methods | N-Cyclone | N-Anticyclone | N-All |
| :-----: | :-------: | :-----------: | :---: |
|   PET   |   4437    |     3737      | 8210  |
| EddyNet |   4502    |     3794      | 8296  |
| PSPNet  |   4614    |     3906      | 8520  |
|  Ours   |   4662    |     3952      | 8614  |



### Others

 Please refer to the citations in this paper(access address: https://doi.org/10.1109/TGRS.2020.3032523 or https://ieeexplore.ieee.org/document/9247537) for SCSE-Eddy dataset. 
 We will continuously update for researchers.


##### We look forward to new breakthroughs for scholars in the future !
