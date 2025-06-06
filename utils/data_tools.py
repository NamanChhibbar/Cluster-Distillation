import numpy as np
import scipy.cluster.hierarchy as sch 

from text_utils import TextPreprocessor,SegmenterEmbedder,BiMapping
from helpers import get_device

def L2(x, y):
    """Calculates the L2 (Euclidean) distance between two numpy arrays."""
    return np.linalg.norm(x - y)

class ClusterModel:
    """
    Performs hierarchical clustering on input embeddings.
    """
    def __init__(
        self,
        dist = L2,
        num_clusters: int = 10,
    ):
        """
        Initializes the ClusterModel.

        Args:
            dist (callable): A function to calculate the distance between two embeddings.
                             Defaults to L2 (Euclidean distance).
            num_clusters (int): The desired number of clusters.
        """
        self.num_clusters = num_clusters
        self.dist = dist

    def fit(self, embeddings: np.ndarray) -> dict:
        """
        Clusters the provided embeddings using hierarchical clustering.

        Args:
            embeddings (np.ndarray): A 2D numpy array where each row is an embedding.

        Returns:
            dict: A dictionary where keys are cluster IDs (integers starting from 1)
                  and values are dictionaries. Each inner dictionary maps
                  original embedding indexes to their corresponding embedding vectors.
            Example:
            {
                1: {0: embedding_0, 5: embedding_5},
                2: {1: embedding_1, 3: embedding_3, 7: embedding_7},
                ...
            }
        """
        # Perform hierarchical clustering using single linkage and force num_clusters.
        # fclusterdata returns cluster IDs for each observation.
        ids = sch.fclusterdata(embeddings, self.num_clusters, criterion='maxclust', method='single', metric=self.dist)

        clusters = {}
        for index, embedding in enumerate(embeddings):
            cluster_id = ids[index]
            if cluster_id in clusters:
                clusters[cluster_id][index] = embedding
            else:
                clusters[cluster_id] = {index: embedding}
        return clusters


class FilterModel:
    """
    Selects a representative embedding (medoid-like) from each cluster.
    The representative is the embedding that has the minimum total distance
    to all other embeddings within its cluster.
    """
    def __init__(
        self,
        dist = L2,
    ):
        """
        Initializes the FilterModel.

        Args:
            dist (callable): A function to calculate the distance between two embeddings.
                             Defaults to L2 (Euclidean distance).
        """
        self.dist = dist

    def fit(self, clusters: dict) -> list:
        """
        Finds the representative (medoid-like) for each cluster.

        Args:
            clusters (dict): A dictionary of clusters, as returned by ClusterModel.fit().
                             {cluster_id: {embedding_index: embedding_vector, ...}, ...}

        Returns:
            list: A sorted list of original embedding indexes that are
                  chosen as representatives for each cluster.
        """
        reps = []
        for cluster_id, embeddings_dict in clusters.items():
            current_representative_index = -1
            min_total_spread = float('inf') # Initialize with infinity

            for index, candidate_embedding in embeddings_dict.items():
                total_spread = 0
                for neighbour_index, neighbour_embedding in embeddings_dict.items():
                    # No need to calculate distance to itself, or it will be 0
                    # For a small performance gain, could add: if index != neighbour_index:
                    total_spread += self.dist(candidate_embedding, neighbour_embedding)

                # Check if this candidate is better than the current best
                if total_spread < min_total_spread:
                    min_total_spread = total_spread
                    current_representative_index = index

            if current_representative_index != -1: # Ensure a representative was found (cluster not empty)
                reps.append(current_representative_index)

        reps.sort() # Sort the representative indexes for consistent output
        return reps


class Shortener:
    """
    A class to preprocess text, segment it, embed segments, cluster them,
    and then select representative segments to shorten the original text.
    """
    def __init__(self):
        """
        Initializes the Shortener with a TextPreprocessor and SegmenterEmbedder.
        """
        self.preprocessor = TextPreprocessor()
        # Corrected typo: SegmentEmbedder -> SegmenterEmbedder
        self.segmenter_embedder = SegmenterEmbedder(device=get_device())

    def fit(self, text: str, num_clusters_for_shortening: int = 100) -> list:
        """
        Shortens the input text by identifying and returning representative segments.

        Args:
            text (str): The input text to be shortened.
            num_clusters_for_shortening (int): The target number of clusters to form,
                                               which directly influences the number of
                                               representative segments.

        Returns:
            list: A list of strings, where each string is a representative segment
                  from the original text.
        """
        preprocessed_text = self.preprocessor(text)

        segments, embeddings = self.segmenter_embedder(preprocessed_text) # Corrected call

        # Ensure embeddings is a numpy array for indexing later
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        mapping = BiMapping(segments, embeddings)

        # Corrected parameter name: num_cluster -> num_clusters
        cluster_model = ClusterModel(num_clusters=num_clusters_for_shortening)
        clusters = cluster_model.fit(embeddings)

        filter_model = FilterModel()
        # Corrected object call: fm.fit -> filter_model.fit
        embedding_indexes = filter_model.fit(clusters)

        # Ensure embeddings is indexable (should be if it's a numpy array)
        representative_embeddings = embeddings[embedding_indexes]

        # Use the BiMapping to get segments from representative embeddings
        # Note: BiMapping expects hashable keys (tuples for numpy arrays)
        representative_segments = [mapping[emb] for emb in representative_embeddings]
        return representative_segments