from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy import stats
class data_helper:
    def __init__(self):
        self.data_dir = 'data'
        pass
    
    ##
    ## Original Data
    ##
    def get_pima_data(self):
        filename = "./data/pima-indians-diabetes.data"
        dataset = np.genfromtxt(filename, delimiter=",")
        x = dataset[1:-1, :-1]
        y = dataset[1:-1, -1]
        x = stats.zscore(x, axis=1, ddof=1)
        return train_test_split(x,
                                y,
                                test_size=0.7,
                                random_state=0)

    def get_spam_data(self):
        filename = "./data/spambase.data"
        dataset = np.genfromtxt(filename, delimiter=",")
        x = dataset[1:-1, :-1]
        y = dataset[1:-1, -1]
        x = stats.zscore(x, axis=1, ddof=1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        y_train = y_train.reshape(-1, 1).astype(float)
        y_test = y_test.reshape(-1, 1).astype(float)
        return X_train, X_test, y_train, y_test
    
    ##
    ## Dimension Reduction Data
    ##
    def get_spam_data_lda_best(self):
        filename = './' + self.data_dir + '/spam_lda_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_lda_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_lda_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_lda_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    
    def get_spam_data_pca_best(self):
        filename = './' + self.data_dir + '/spam_pca_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_pca_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_pca_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_pca_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    
    def get_spam_data_rp_best(self):
        filename = './' + self.data_dir + '/spam_rp_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_rp_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_rp_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_rp_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
        
        
    def get_spam_data_ica_best(self):
        filename = './' + self.data_dir + '/spam_ica_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_ica_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_ica_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_ica_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
        
        
    def get_pima_data_lda_best(self):
        filename = './' + self.data_dir + '/pima_lda_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_lda_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_lda_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_lda_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    
    def get_pima_data_pca_best(self):
        filename = './' + self.data_dir + '/pima_pca_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_pca_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_pca_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_pca_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    
    def get_pima_data_rp_best(self):
        filename = './' + self.data_dir + '/pima_rp_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_rp_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_rp_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_rp_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
        
        
    def get_pima_data_ica_best(self):
        filename = './' + self.data_dir + '/pima_ica_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_ica_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_ica_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/pima_ica_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
        
    ##
    ## Dimension Reduction plus Clustering Data
    ##
    def get_spam_data_kmeans_pca_best(self):
        filename = './' + self.data_dir + '/spam_kmeans_pca_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_pca_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_pca_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_pca_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    
    def get_spam_data_gmm_pca_best(self):
        filename = './' + self.data_dir + '/spam_gmm_pca_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_pca_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_pca_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_pca_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_spam_data_kmeans_ica_best(self):
        filename = './' + self.data_dir + '/spam_kmeans_ica_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_ica_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_ica_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_ica_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_spam_data_gmm_ica_best(self):
        filename = './' + self.data_dir + '/spam_gmm_ica_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_ica_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_ica_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_ica_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_spam_data_kmeans_rp_best(self):
        filename = './' + self.data_dir + '/spam_kmeans_rp_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_rp_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_rp_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_rp_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_spam_data_gmm_rp_best(self):
        filename = './' + self.data_dir + '/spam_gmm_rp_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_rp_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_rp_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_rp_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_spam_data_kmeans_lda_best(self):
        filename = './' + self.data_dir + '/spam_kmeans_lda_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_lda_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_lda_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_kmeans_lda_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
    def get_spam_data_gmm_lda_best(self):
        filename = './' + self.data_dir + '/spam_gmm_lda_x_train.txt'
        df1 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_lda_x_test.txt'
        df2 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_lda_y_train.txt'
        df3 = pd.read_csv(filename, sep=',')
        
        filename = './' + self.data_dir + '/spam_gmm_lda_y_test.txt'
        df4 = pd.read_csv(filename, sep=',')

        X_train = df1.values
        X_test = df2.values
        y_train = df3.values
        y_test = df4.values
        
        return X_train, X_test, y_train, y_test
    
if __name__== '__main__':
    print(1)
