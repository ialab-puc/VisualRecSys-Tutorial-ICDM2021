class VisualSimilarityHandler:
    def __init__(self, cluster_ids, embeddings):
        self._cluster_ids = cluster_ids
        self._cosineSimCache = dict()
        self.count = 0
        # store embeddings with l2 normalization
        from numpy.linalg import norm
        from numpy import dot
        self._embeddings = embeddings / norm(embeddings, axis=1).reshape((-1,1))
        self.dot = dot
        
    def same(self,i,j):
        if self._cluster_ids[i] != self._cluster_ids[j]:
            return False        
        if abs(self.similarity(i,j) - 1.) < 1e-7:
            self.count += 1
            return True
        return False
    
    def similarity(self,i,j):
        if i > j:
            i, j = j, i
        k = (i,j)
        try:
            sim = self._cosineSimCache[k]
        except KeyError:
            sim = self._cosineSimCache[k] = self.dot(self._embeddings[i], self._embeddings[j])
        return sim
    
    def validate_triple(self, q, p, n, margin=0.05):
        cq = self._cluster_ids[q]
        cp = self._cluster_ids[p]
        cn = self._cluster_ids[n]
        if cq == cp and cq != cn:
            return True
        if cq == cn and cq != cp:
            return False
        if self.similarity(q,p) > self.similarity(q,n) + margin:
            return True
        return False
    
class HybridScorer:

    def __init__(self, vissim_handler, artists, artist_boost):
        self.vissim_handler = vissim_handler
        self.artists = artists
        self.artist_boost = artist_boost
        self.score_cache = dict()
        
    def simfunc(self, i, j):
        sim = self.vissim_handler.similarity(i, j)
        ai = self.artists[i] # .get(i, -1)
        if ai == -1: return sim        
        aj = self.artists[j]
        if ai == aj: sim += self.artist_boost
        return sim
    
    def get_score(self, u, profile, i):
        key = (u,i)
        try:
            return self.score_cache[key]
        except KeyError:
            score = sum(self.simfunc(i,j) for j in profile) / len(profile)
            self.score_cache[key] = score
            return score