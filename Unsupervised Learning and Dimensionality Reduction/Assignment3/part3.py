import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import itertools

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA, FastICA, RandomizedPCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import *

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from timeit import default_timer as time

from data_helper import *
from plot_helper import *
from part1 import *

'''
3. Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
'''
class part3():
    def __init__(self):
        self.save_dir = 'data'
        self.out_dir = 'output_part3'
        self.part1 = part1()
        self.part1.out_dir = self.out_dir
        self.time_filename = './' + self.out_dir + '/time.txt'
    
    def run(self):
        print('Running part 3')
        with open(self.time_filename, 'w') as text_file:
            
            t0 = time()
            self.kmeans_pca_spam()
            text_file.write('kmeans_pca_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_pca_pima()
            text_file.write('kmeans_pca_pima: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_ica_spam()
            text_file.write('kmeans_ica_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_ica_pima()
            text_file.write('kmeans_ica_pima: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_rp_spam()
            text_file.write('kmeans_rp_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_rp_pima()
            text_file.write('kmeans_rp_pima: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_lda_spam()
            text_file.write('kmeans_lda_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_lda_pima()
            text_file.write('kmeans_lda_pima: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_pca_spam()
            text_file.write('gmm_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_pca_pima()
            text_file.write('gmm_pima: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_ica_spam()
            text_file.write('gmm_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_ica_pima()
            text_file.write('gmm_pima: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_rp_spam()
            text_file.write('gmm_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_rp_pima()
            text_file.write('gmm_pima: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_lda_spam()
            text_file.write('gmm_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_lda_pima()
            text_file.write('gmm_pima: %0.3f seconds\n' % (time() - t0))
            
            ##
            ## Generate files for best
            ##
            
            # only need wine for parts 4 and 5
            self.best_pca_cluster_spam()
            self.best_ica_cluster_spam()
            self.best_rp_cluster_spam()
            self.best_lda_cluster_spam()
            
            

    def best_pca_cluster_spam(self):
        dh = data_helper()
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_pca_best()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## K-Means
        ##
        km = KMeans(n_clusters=3, algorithm='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/spam_kmeans_pca_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_pca_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_pca_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_pca_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        ##
        ## GMM
        ##
        gmm = GaussianMixture(n_components=3, covariance_type='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/spam_gmm_pca_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_pca_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_pca_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_pca_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
    def best_ica_cluster_spam(self):
        dh = data_helper()
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_ica_best()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## K-Means
        ##
        km = KMeans(n_clusters=3, algorithm='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/spam_kmeans_ica_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_ica_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_ica_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_ica_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        ##
        ## GMM
        ##
        gmm = GaussianMixture(n_components=4, covariance_type='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/spam_gmm_ica_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_ica_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_ica_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_ica_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
    def best_rp_cluster_spam(self):
        dh = data_helper()
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_rp_best()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## K-Means
        ##
        km = KMeans(n_clusters=5, algorithm='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/spam_kmeans_rp_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_rp_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_rp_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_rp_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        ##
        ## GMM
        ##
        gmm = GaussianMixture(n_components=3, covariance_type='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/spam_gmm_rp_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_rp_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_rp_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_rp_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
    
    def best_lda_cluster_spam(self):
        dh = data_helper()
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_lda_best()
        
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        ##
        ## K-Means
        ##
        km = KMeans(n_clusters=4, algorithm='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/spam_kmeans_lda_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_lda_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_lda_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_kmeans_lda_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        ##
        ## GMM
        ##
        gmm = GaussianMixture(n_components=4, covariance_type='full')
        X_train_transformed = km.fit_transform(X_train_scl)
        X_test_transformed = km.transform(X_test_scl)
        
        # save
        filename = './' + self.save_dir + '/spam_gmm_lda_x_train.txt'
        pd.DataFrame(X_train_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_lda_x_test.txt'
        pd.DataFrame(X_test_transformed).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_lda_y_train.txt'
        pd.DataFrame(y_train).to_csv(filename, header=False, index=False)
        
        filename = './' + self.save_dir + '/spam_gmm_lda_y_test.txt'
        pd.DataFrame(y_test).to_csv(filename, header=False, index=False)
        
        
    def kmeans_pca_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_pca_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Spam', 20, 'K-Means PCA')
    
    def kmeans_pca_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data_pca_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Pima', 20, 'K-Means PCA')
        
    def kmeans_ica_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_ica_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Spam', 20, 'K-Means ICA')
    
    def kmeans_ica_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data_ica_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Pima', 20, 'K-Means ICA')
        
    def kmeans_rp_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_rp_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Spam', 20, 'K-Means RP')
    
    def kmeans_rp_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data_rp_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Pima', 20, 'K-Means RP')
        
    def kmeans_lda_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_lda_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Spam', 20, 'K-Means LDA')
    
    def kmeans_lda_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data_lda_best()
        self.part1.kmeans_analysis(X_train, X_test, y_train, y_test, 'Pima', 20, 'K-Means LDA')
        
    def gmm_pca_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_pca_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Spam', 20, 'GMM PCA')
    
    def gmm_pca_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data_pca_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Pima', 20, 'GMM PCA')
        
    def gmm_ica_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_ica_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Spam', 20, 'GMM ICA')
    
    def gmm_ica_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data_ica_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Pima', 20, 'GMM ICA')
        
    def gmm_rp_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_rp_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Spam', 20, 'GMM RP')
    
    def gmm_rp_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data_rp_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Pima', 20, 'GMM RP')
        
    def gmm_lda_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data_lda_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Spam', 20, 'GMM LDA')
    
    def gmm_lda_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data_lda_best()
        self.part1.gmm_analysis(X_train, X_test, y_train, y_test, 'Pima', 20, 'GMM LDA')

    
def main():    
    p = part3()
    p.run()
    
if __name__== '__main__':
    main()
    
