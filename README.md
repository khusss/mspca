# mspca (MSPCA)
Multiscale Principal Component Analysis.

Multiscale PCA (MSPCA) combines the ability of PCA to extract the crosscorrelation or relationship
between the variables, with that of orthonormal wavelets to separate deterministic features
from stochastic processes and approximately decorrelate the autocorrelation among the measurements[1].

<img src="https://user-images.githubusercontent.com/28721422/111423028-0280a800-8733-11eb-8a68-4726130eb542.PNG" width="61%">

*Fig 1. Schematic illustration of MSPCA model[2].*


<img src="https://user-images.githubusercontent.com/28721422/111423035-04e30200-8733-11eb-92b7-bf08f452ef56.PNG" width="61%">

*Fig 2. Schematic diagram for multiscale representation of data[2].*


*******
#### References
[1] Bhavik R. Bakshi, Multiscale PCA with Application to Multivariate Statistical Process Monitoring, The Ohio State University, 1998.

[2] M. Ziyan Sheriff, Majdi Mansouri, M. Nazmul Karim, Hazem Nounou, Fault detection using multiscale PCA-based moving window GLRT, Journal of Process Control, 2017.

# Installation
#### Dependencies
mspca requires:

+ Python >= 3.7
+ PyWavelets == 1.0.3
+ numpy == 1.19.5
+ pandas == 0.25.1


#### Pip
The easiest way to install mspca is using 'pip'

    pip install mspca

# Example

    from mspca import mspca

    mymodel = mspca.MultiscalePCA()
    X_pred = mymodel.fit_transform(X, wavelet_func='db4', threshold=0.3)

![example1](https://user-images.githubusercontent.com/28721422/111422652-5939b200-8732-11eb-9d92-e966191e2b72.PNG)
![example2](https://user-images.githubusercontent.com/28721422/111422673-62c31a00-8732-11eb-9ff2-b74824fc62cb.PNG)
![example3](https://user-images.githubusercontent.com/28721422/111423017-feed2100-8732-11eb-8c11-acf498dffef0.PNG)

# Contact us
Heeyu Kim / khudd@naver.com

Kyuhan Seok / asdm159@naver.com
