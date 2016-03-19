import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import coo_matrix
import glob
from itertools import izip
from scipy.sparse import find
import binascii

class LSH_sparse:
    def __init__(self,file_list):
        self.file_list = file_list
        hv = HashingVectorizer(input='filename',binary=True, \
                               n_features=(2 ** 18),norm=None, \
                               stop_words='english',\
                               ngram_range=(2, 2), \
                               decode_error='ignore')
        self.sparse_hashed_article  = hv.transform(self.file_list)
        self.total_articles = self.sparse_hashed_article.shape[0]
        print self.total_articles

    def set_hashes(self, num_hashes):
        self.total_hashes = num_hashes

    def set_M(self, M):
        self.M = M

    def min_hash_articles(self):
        a = np.random.randint(0,self.M,size=self.total_hashes)
        b = np.random.randint(0,self.M,size=self.total_hashes)
        self.hashed_articles = np.zeros([self.total_articles,self.total_hashes])
        for art_index in range(0, self.total_articles):
            # get the article
            article = self.sparse_hashed_article[art_index]
            # get the index of all non zero elements of the article
            shingle_ID = find(article)[1]
            for i in range(0, self.total_hashes):
                self.hashed_articles[art_index,i] = \
                                self.find_min_hash(shingle_ID,a[i],b[i])

    def find_min_hash(self,arr,a,b):
        return np.min((a * arr + b) % self.M)

    def get_min_hash_matrix(self):
        return self.hashed_articles


    def bucket_bands(self, band_size, hash_per_band):
        self.band_size = band_size
        self.hash_per_band = hash_per_band
        # initialize the list_of_list_of_sets
        self.list_of_dictionaries = np.zeros(self.band_size,dtype=dict)
        for band in range(0,self.band_size):
            self.list_of_dictionaries[band] = dict()


        for article_id in range(0,self.total_articles):
            row = self.hashed_articles[article_id]
            tuple_list = [tuple(row[i:i + self.hash_per_band]) \
                            for i in xrange(0, len(row), self.hash_per_band)]
            # hash them
            for band in range(0,self.band_size):
                if tuple_list[band] not in self.list_of_dictionaries[band]:
                    self.list_of_dictionaries[band][tuple_list[band]] = \
                                                                [article_id]
                else:
                    self.list_of_dictionaries[band][tuple_list[band]].append(\
                                                                article_id)


    def get_best_30(self):
        global_tuple_dict = dict()
        for band_id in range(0, self.band_size):
            for all_pairs in self.list_of_dictionaries[band_id].values():
                print all_pairs
                for i in range(0,len(all_pairs)):
                    for j in range(i + 1, len(all_pairs)):
                        neighbor_tuple = tuple((all_pairs[i],all_pairs[j]))
                        if neighbor_tuple not in global_tuple_dict:
                            # get the jaccard similarity
                            jaccard_sim =self.get_estimated_jaccard_similarity(\
                                self.hashed_articles[all_pairs[i]],\
                                self.hashed_articles[all_pairs[j]])
                            global_tuple_dict[neighbor_tuple] = jaccard_sim

        # get top 30 elements
        self.list_of_tuples = []
        for key in global_tuple_dict.iterkeys():
            values = global_tuple_dict[key]
            tuple_final = tuple((values,key[0],key[1]))
            self.list_of_tuples.append(tuple_final)

        self.list_of_tuples.sort(reverse=True)
        print self.list_of_tuples
        self.list_of_tuples = self.list_of_tuples[0:30]

    def get_estimated_jaccard_similarity(self,root_article,comparison_article):
        intersection_count = 0
        for i in range(0,self.total_hashes):
            if(root_article[i] == comparison_article[i]):
                intersection_count += 1
        return float(intersection_count)/self.total_hashes

    def save_best_pairs(self):
        rows = []
        for article_pairs in self.list_of_tuples:
            row = {'name1': self.file_list[article_pairs[1]], \
                   'name2':self.file_list[article_pairs[2]], \
                   'id1': article_pairs[1],\
                   'id2': article_pairs[2],\
                   'similarity': article_pairs[0]}
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv('GutenbergMinHash.csv')


def main():
    ######################Collaborative Filtering##############################

    ######## Load the data and hash it to shingle
    file_list = glob.glob('Gutenberg/txt/*.txt')
    print 'Getting the shingle hashed sparse matrix.....'
    lsh_sparse = LSH_sparse(file_list)
    print 'Obtained the hashed shingle matrix. Moving froward...'

    ######### Minhash the article
    total_hashes = 90
    lsh_sparse.set_hashes(total_hashes)
    M = 262147
    lsh_sparse.set_M(M)
    print 'Min-hashing the shingle hashed document'
    lsh_sparse.min_hash_articles()
    print 'min hash complete. Moving forward'

    ######### Finding close hash articles using LSH with band = 30 and r = 3
    band_size = 30
    hash_per_band = 3
    print 'Finding the set of all close articles....'
    lsh_sparse.bucket_bands(band_size,hash_per_band)
    print 'Obtained the set of all close articles for b = 30, r = 3'

    ######### Get the best 30 articles pairs
    ## get the index of the best 30 articles
    lsh_sparse.get_best_30()

    ######### Save it in the pandas
    lsh_sparse.save_best_pairs()

if __name__ == "__main__":
    main()
