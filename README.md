# mspca (MSPCA)
Multiscale Principal Component Analysis.

Multiscale PCA (MSPCA) combines the ability of PCA to extract the crosscorrelation or relationship
between the variables, with that of orthonormal wavelets to separate deterministic features
from stochastic processes and approximately decorrelate the autocorrelation among the measurements[1].

![MSPCA_MODEL_IMAGE](https://user-images.githubusercontent.com/28721422/111423028-0280a800-8733-11eb-8a68-4726130eb542.PNG)

*Fig 1. Schematic illustration of MSPCA model[2].*


![MSPCA_RESULT](./img/mspca_signal.png)

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

    import mspca

    mymodel = mspca.MultiscalePCA()
    x_pred = mymodel.fit_transform(X, wavelet_func='db4', threshold=0.3)

![example1](./img/example1.png)
![example2](./img/example2.png)
![example3](./img/example3.png)

# Contact us
Heeyu Kim / khudd@naver.com

Kyuhan Seok / asdm159@naver.com
