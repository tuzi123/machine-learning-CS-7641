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

'''
1. Run the clustering algorithms on the data sets and describe what you see.
'''
class part1():
    def __init__(self):
        self.out_dir = 'output_part1'
        self.time_filename = './' + self.out_dir + '/time.txt'
    
    def run(self):
        print('Running part 1')
        with open(self.time_filename, 'w') as text_file:
            
            t0 = time()
            self.gmm_spam()
            text_file.write('gmm_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.gmm_pima()
            text_file.write('gmm_pima: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_spam()
            text_file.write('kmeans_spam: %0.3f seconds\n' % (time() - t0))
            
            t0 = time()
            self.kmeans_pima()
            text_file.write('kmeans_pima: %0.3f seconds\n' % (time() - t0))

            

    def gmm_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()
        self.gmm_analysis(X_train, X_test, y_train, y_test, 'Spam', 30)
    
    def gmm_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()
        self.gmm_analysis(X_train, X_test, y_train, y_test, 'Pima', 30)
        
    def kmeans_spam(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_spam_data()
        self.kmeans_analysis(X_train, X_test, y_train, y_test, 'Spam', 20)
    
    def kmeans_pima(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test = dh.get_pima_data()
        self.kmeans_analysis(X_train, X_test, y_train, y_test, 'Pima', 20)
    
    # source: https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch11/ch11.ipynb
    def silhouette_plot(self, X, X_predicted, title, filename):
        plt.clf()
        plt.cla()
        
        cluster_labels = np.unique(X_predicted)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = silhouette_samples(X, X_predicted, metric='euclidean')
        y_ax_lower, y_ax_upper = 0, 0
        
        color=iter(cm.viridis(np.linspace(0,1,cluster_labels.shape[0])))
           
        yticks = []
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[X_predicted == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=next(color))
        
            yticks.append((y_ax_lower + y_ax_upper) / 2.)
            y_ax_lower += len(c_silhouette_vals)
            
        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg, color="red", linestyle="--") 
        
        plt.yticks(yticks, cluster_labels + 1)
        plt.ylabel('Cluster')
        plt.xlabel('Silhouette Coefficient')
        
        plt.title(title)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')

                    
    def kmeans_analysis(self, X_train, X_test, y_train, y_test, data_set_name, max_clusters, analysis_name='K-Means'):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        km_inertias = []
        km_completeness_score = []
        km_homogeneity_score = []
        km_measure_score = []
        km_adjusted_rand_score = []
        km_adjusted_mutual_info_score = []
        
        cluster_range = np.arange(2, max_clusters+1, 1)
        for k in cluster_range:
            print('K Clusters: ', k)
            ##
            ## KMeans
            ##
            km = KMeans(n_clusters=k, algorithm='full', n_jobs=-1)
            km.fit(X_train_scl)
            
            # inertia is the sum of distances from each point to its center   
            km_inertias.append(km.inertia_)
            
            # metrics
            y_train_score = y_train.reshape(y_train.shape[0],)
            
            km_homogeneity_score.append(homogeneity_score(y_train_score, km.labels_))
            km_completeness_score.append(completeness_score(y_train_score, km.labels_))
            km_measure_score.append(v_measure_score(y_train_score, km.labels_))
            km_adjusted_rand_score.append(adjusted_rand_score(y_train_score, km.labels_))
            km_adjusted_mutual_info_score.append(adjusted_mutual_info_score(y_train_score, km.labels_))
            
            ##
            ## Silhouette Plot
            ##
            title = 'Silhouette Plot (' + analysis_name + ', k=' + str(k) + ') for ' + data_set_name
            name = data_set_name.lower() + '_' + analysis_name.lower() + '_silhouette_' + str(k)
            filename = './' + self.out_dir + '/' + name + '.png'
            
            self.silhouette_plot(X_train_scl, km.labels_, title, filename)
            
        ##
        ## Plots
        ##
        ph = plot_helper()
        
        ##
        ## Elbow Plot
        ##
        title = 'Elbow Plot (' + analysis_name + ') for ' + data_set_name
        name = data_set_name.lower() + '_' + analysis_name.lower() + '_elbow'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        # line to help visualize the elbow
        lin = ph.extended_line_from_first_two_points(km_inertias, 0, 2)
        
        ph.plot_series(cluster_range,
                    [km_inertias, lin],
                    [None, None],
                    ['inertia', 'projected'],
                    cm.viridis(np.linspace(0, 1, 2)),
                    ['o', ''],
                    title,
                    'Number of Clusters',
                    'Inertia',
                    filename)
        
        ##
        ## Score Plot
        ##
        title = 'Score Summary Plot (' + analysis_name + ') for ' + data_set_name
        name = data_set_name.lower() + '_' + analysis_name.lower() + '_score'
        filename = './' + self.out_dir + '/' + name + '.png'
                    
        ph.plot_series(cluster_range,
                    [km_homogeneity_score, km_completeness_score, km_measure_score, km_adjusted_rand_score, km_adjusted_mutual_info_score],
                    [None, None, None, None, None, None],
                    ['homogeneity', 'completeness', 'measure', 'adjusted_rand', 'adjusted_mutual_info'],
                    cm.viridis(np.linspace(0, 1, 5)),
                    ['o', '^', 'v', '>', '<', '1'],
                    title,
                    'Number of Clusters',
                    'Score',
                    filename)
        
    def gmm_analysis(self, X_train, X_test, y_train, y_test, data_set_name, max_clusters, analysis_name='GMM'):
        scl = RobustScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)
        
        em_bic = []
        em_aic = []
        em_completeness_score = []
        em_homogeneity_score = []
        em_measure_score = []
        em_adjusted_rand_score = []
        em_adjusted_mutual_info_score = []
        
        cluster_range = np.arange(2, max_clusters+1, 1)
        for k in cluster_range:
            print('K Clusters: ', k)
            
            ##
            ## Expectation Maximization
            ##
            em = GaussianMixture(n_components=k, covariance_type='full')
            em.fit(X_train_scl)
            em_pred = em.predict(X_train_scl)
            
            em_bic.append(em.bic(X_train_scl))
            em_aic.append(em.aic(X_train_scl))        
        
            # metrics
            y_train_score = y_train.reshape(y_train.shape[0],)
            
            em_homogeneity_score.append(homogeneity_score(y_train_score, em_pred))
            em_completeness_score.append(completeness_score(y_train_score, em_pred))
            em_measure_score.append(v_measure_score(y_train_score, em_pred))
            em_adjusted_rand_score.append(adjusted_rand_score(y_train_score, em_pred))
            em_adjusted_mutual_info_score.append(adjusted_mutual_info_score(y_train_score, em_pred))
            
        
        ##
        ## Plots
        ##
        ph = plot_helper()
        
        ##
        ## BIC/AIC Plot
        ##
        title = 'Information Criterion Plot (' + analysis_name + ') for ' + data_set_name
        name = data_set_name.lower() + '_' + analysis_name.lower() + '_ic'
        filename = './' + self.out_dir + '/' + name + '.png'
        
        ph.plot_series(cluster_range,
                    [em_bic, em_aic],
                    [None, None],
                    ['bic', 'aic'],
                    cm.viridis(np.linspace(0, 1, 2)),
                    ['o', '*'],
                    title,
                    'Number of Clusters',
                    'Information Criterion',
                    filename)
        
        ##
        ## Score Plot
        ##
        title = 'Score Summary Plot (' + analysis_name + ') for ' + data_set_name
        name = data_set_name.lower() + '_' + analysis_name.lower() + '_score'
        filename = './' + self.out_dir + '/' + name + '.png'
                    
        ph.plot_series(cluster_range,
                    [em_homogeneity_score, em_completeness_score, em_measure_score, em_adjusted_rand_score, em_adjusted_mutual_info_score],
                    [None, None, None, None, None, None],
                    ['homogeneity', 'completeness', 'measure', 'adjusted_rand', 'adjusted_mutual_info'],
                    cm.viridis(np.linspace(0, 1, 5)),
                    ['o', '^', 'v', '>', '<', '1'],
                    title,
                    'Number of Clusters',
                    'Score',
                    filename)
        
def main():
    p = part1()
    p.run()
    
    
if __name__== '__main__':
    main()
    
