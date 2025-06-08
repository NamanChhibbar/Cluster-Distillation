import numpy as np
import scipy.cluster.hierarchy as sch
import torch

from .text_utils import TextPreprocessor, SegmenterEmbedder, BiMapping


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
    ):
        """
        Initializes the ClusterModel.

        Args:
            dist (callable): A function to calculate the distance between two embeddings.
                             Defaults to L2 (Euclidean distance).
            num_clusters (int): The desired number of clusters.
        """
        self.dist = dist

    def fit(self, embeddings: np.ndarray, num_clusters : int = 10) -> dict:
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
        ids = sch.fclusterdata(embeddings, num_clusters, criterion='maxclust', method='single', metric=self.dist)

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
        for _, embeddings_dict in clusters.items():
            current_representative_index = -1
            min_total_spread = float('inf')

            for index, candidate_embedding in embeddings_dict.items():
                total_spread = 0
                for _, neighbour_embedding in embeddings_dict.items():
                    total_spread += self.dist(candidate_embedding, neighbour_embedding)

                if total_spread < min_total_spread:
                    min_total_spread = total_spread
                    current_representative_index = index

            if current_representative_index != -1: 
                reps.append(current_representative_index)

        reps.sort()
        return reps


class Shortener:
    """
    A class to preprocess text, segment it, embed segments, cluster them,
    and then select representative segments to shorten the original text.
    """
    def __init__(self, device: str | torch.device = 'cpu'):
        """
        Initializes the Shortener with a TextPreprocessor and SegmenterEmbedder.
        """
        self.preprocessor = TextPreprocessor()
        self.segmenter_embedder = SegmenterEmbedder(device=device)
        self.cluster_model = ClusterModel()
        self.filter_model = FilterModel()   

    def fit(self, text: str, num_clusters: int) -> list[str]:
        """
        Shortens the input text by identifying and returning representative segments.

        Args:
            text (str): The input text to be shortened.
            num_clusters (int): The target number of clusters to form, which directly influences the number of representative segments.

        Returns:
            list: A list of strings, where each string is a representative segment from the original text.
        """
        preprocessed_text = self.preprocessor(text)
        segments, embeddings = self.segmenter_embedder(preprocessed_text)
        mapping = BiMapping(segments, embeddings)
        clusters = self.cluster_model.fit(embeddings, num_clusters=num_clusters)
        embedding_indexes = self.filter_model.fit(clusters)
        representative_embeddings = embeddings[embedding_indexes]
        representative_segments = [mapping[emb] for emb in representative_embeddings]
        return representative_segments
