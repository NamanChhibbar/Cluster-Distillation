import numpy as np
import scipy.cluster.hierarchy as sch 

def L2(x,y):
    return np.linalg.norm(x - y)

class cluster_model:
    def __init__(
        self,
        dist = L2,
        num_clusters = 10,
    ):
        self.num_clusters = num_clusters
        self.dist = dist
    
    def fit(self,embeddings):
        '''        
        Return
        {
            cluster_id : {
                embedding_index : embedding,
                ..
            },
            ..
        }
        '''
        ids = sch.fclusterdata(embeddings, self.num_clusters, criterion='maxclust', method = 'single', metric = self.dist)
        clusters = {}
        for index,embedding in enumerate(embeddings):
            if ids[index] in clusters:
                clusters[ids[index]][index] = embedding
            else:
                clusters[ids[index]] = {index : embedding}
        return clusters


class filter_model:
    def __init__(
        self,
        dist = L2,
    ):
        self.dist = dist
        
    def fit(self, clusters): 
        '''
        Return
        list of index of representatives
        '''
        reps = []
        for cluster_id, embeddings_dict in clusters.items():
            current = -1
            min_total_spread = -1
            
            for index, candidate in embeddings_dict.items():
                total_spread = 0
                for neighbour in embeddings_dict.values():
                    total_spread += self.dist(candidate,neighbour)

                if(current == -1 or total_spread < min_total_spread):
                    current = index
                    min_total_spread = total_spread
            
            reps.append(current)
        
        reps.sort()
        return reps
                  
                