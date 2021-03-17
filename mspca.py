import pywt  # 1.0.3
import numpy as np  # 1.19.5
import pandas as pd  # 0.25.1
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MultiscalePCA:
    """
    - Multiscale Principal Component with Wavelet Decomposition
    - Return estimate of X

    *Example
        import mspca

        mymodel = mspca.MultiscalePCA()
        x_pred = mymodel.fit_transform(X, wavelet_func='db4', threshold=0.3)
    """

    def __init__(self):
        self.x_norm = None
        self.fit_bool = False


    def _split_coef(self, train_data, wavelet_func):
        self.w = pywt.Wavelet(wavelet_func)

        # Wavelet decomposition
        temp_coef = pywt.wavedec(train_data[:, 0], self.w)

        # maxlev = pywt.dwt_max_level(len(x_norm), w.dec_len)

        self.coef_num = len(temp_coef)
        self.x_var_num = train_data.shape[1]
        a_coef_list = []

        for i in range(1, self.coef_num):
            globals()['D{}'.format(i)] = []

        for i in range(self.x_var_num):
            coeffs = pywt.wavedec(train_data[:, i], self.w)

            # Add Approximation Coefficient
            a_coef_list.append(coeffs[0])

            # Add Detailed Coefficient
            for j in range(1, self.coef_num):
                tmp = globals()['D{}'.format(j)]
                tmp.append(coeffs[j])
                globals()['D{}'.format(j)] = tmp

        a_df = pd.DataFrame(a_coef_list)
        a_df = a_df.fillna(0)
        globals()['D{}'.format(0)] = a_df.T

        for i in range(1, self.coef_num):
            df = pd.DataFrame(globals()['D{}'.format(i)])
            df = df.fillna(0)
            globals()['D{}'.format(i)] = df.T

    def _latent_pca(self):
        # extract latent variable using PCA
        for i in range(self.x_var_num):
            globals()['x{}_coeffs'.format(i)] = []

        # Wavelet Coefficient estimation using PCA
        for i in range(self.coef_num):
            pca = PCA(n_components=0.95)
            t_score = pca.fit_transform(globals()['D{}'.format(i)])
            globals()['D{}_hat'.format(i)] = np.matmul(t_score, pca.components_)

            # Wavelet coefficients 변수별 분배
            for k in range(self.x_var_num):
                globals()['x{}_coeffs'.format(k)].append(globals()['D{}_hat'.format(i)][:, k])

    def _de_coef(self, threshold):
        # Denoising Wavelet Coefficients

        for i in range(self.x_var_num):
            for j in range(1, self.coef_num):
                globals()['x{}_coeffs'.format(i)][j] = pywt.threshold(globals()['x{}_coeffs'.format(i)][j],
                                                                      threshold * max(
                                                                          globals()['x{}_coeffs'.format(i)][j]))
    def _rec_coef(self):
        # Wavelet Reconstruction
        res = []
        for k in range(self.x_var_num):
            rec_data = pywt.waverec(globals()['x{}_coeffs'.format(k)], self.w)
            res.append(rec_data)

        df_res = pd.DataFrame(res).T
        self.out_pca = PCA(n_components=0.95)
        t_score = self.out_pca.fit_transform(df_res)
        result = np.matmul(t_score, self.out_pca.components_)

        return result


    def _rec_pred_coef(self):
        # Wavelet Reconstruction
        res = []
        for k in range(self.x_var_num):
            rec_data = pywt.waverec(globals()['x{}_coeffs'.format(k)], self.w)
            res.append(rec_data)

        df_res = pd.DataFrame(res).T

        t_score = self.out_pca.transform(df_res)
        result = np.matmul(t_score, self.out_pca.components_)

        return result

    def predict(self, test_x, scale=True):
        if self.fit_bool:
            if scale:
                test_data = self.scaler.transform(test_x)
            else:
                test_data = test_x
        else:
            print("Fitting model doesn't' exists")

        self._split_coef(test_data, self.wavelet_func)
        self._latent_pca()
        self._de_coef(self.threshold)
        res = self._rec_pred_coef()

        if scale:
            x_pred = self.scaler.inverse_transform(res)
        else:
            x_pred = result

        return x_pred


    def fit_transform(self, x, wavelet_func='db4', threshold=0.3, scale=True):
        """
        :parameter
            x: Array
                Data with noise
            w: str
                Wavelet function, default='db4'
        :return
            x_hat: Numpy Array
        """
        self.wavelet_func = wavelet_func
        self.threshold = threshold
        if scale:
            self.scaler = StandardScaler()
            self.x_norm = self.scaler.fit_transform(x)
        else:
            self.x_norm = x

        self._split_coef(self.x_norm, wavelet_func)

        self._latent_pca()

        self._de_coef(threshold)

        res = self._rec_coef()

        if scale:
            x_hat = self.scaler.inverse_transform(res)
        else:
            x_hat = result

        self.fit_bool = True

        return x_hat
