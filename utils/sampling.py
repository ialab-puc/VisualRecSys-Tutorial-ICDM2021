import random

import numpy as np
from tqdm.auto import tqdm

from .hashing import pre_hash


class StrategyHandler:

    def __init__(self, interactions, vissimhandler, hybrid_scorer,
                 clustId2artIndexes, cluster_by_idx,
                 artistId2artworkIndexes, artist_by_idx,
                 user_as_items,
                 threshold=0.7, confidence_margin=0.18, max_profile_size=None,
                 ):
        self.interactions = interactions
        self.vissimhandler = vissimhandler
        self.hybrid_scorer = hybrid_scorer
        self.clustId2artIndexes = clustId2artIndexes
        self.cluster_by_idx = cluster_by_idx
        self.artistId2artworkIndexes = artistId2artworkIndexes
        self.artist_by_idx = artist_by_idx
        self.user_as_items = user_as_items
        self.threshold = threshold
        self.confidence_margin = confidence_margin
        self.max_profile_size = max_profile_size

    def __sample_artwork_index(self, idx, n_clusters=100):
        if random.random() <= self.threshold:
            if self.artist_by_idx[idx] == -1 or random.random() <= 0.5:
                j = random.choice(self.clustId2artIndexes[self.cluster_by_idx[idx]])
            else:
                j = random.choice(self.artistId2artworkIndexes[self.artist_by_idx[idx]])
        else:
            c = random.randint(0, n_clusters-1)
            j = random.choice(self.clustId2artIndexes[c])
        return j

    def __sample_artwork_index_smart(self, artists_list, clusters_list, profile_set, n_clusters=100):
        while True:
            if random.random() <= self.threshold:
                if random.random() <= 0.5:
                    a = random.choice(artists_list)
                    i = random.choice(self.artistId2artworkIndexes[a])
                else:
                    c = random.choice(clusters_list)
                    i = random.choice(self.clustId2artIndexes[c])
            else:
                c = random.randint(0, n_clusters-1)
                i = random.choice(self.clustId2artIndexes[c])
            if i not in profile_set:
                return i

    def __sample_artwork_index_naive(self, profile_set):
        while True:
            ni = random.randint(0, len(self.cluster_by_idx) - 1)
            if ni not in profile_set:
                return ni

    def strategy_1(self, samples_per_user, hashes_container):
        # Initialization
        interactions = self.interactions.copy()
        samples = []
        for ui, group in tqdm(interactions.groupby("user_id"), desc="Strategy 1"):
            # Get profile artworks
            full_profile = np.hstack(group["item_id"].values).tolist()
            full_profile_set = set(full_profile)
            n = samples_per_user
            while n > 0:
                # Sample positive and negative items
                pi_index = random.randrange(len(full_profile))
                pi = full_profile[pi_index]
                # Get profile
                if self.max_profile_size:
                    # "pi_index + 1" to include pi in profile
                    profile = full_profile[max(0, pi_index - self.max_profile_size + 1):pi_index + 1]
                else:
                    profile = list(full_profile)
                while True:
                    ni = self.__sample_artwork_index(pi)
                    if ni not in full_profile_set:
                        break
                # Compare visual similarity
                if self.vissimhandler.same(pi, ni):
                    continue
                # Get score from hybrid scorer
                spi = self.hybrid_scorer.get_score(ui, profile, pi)
                sni = self.hybrid_scorer.get_score(ui, profile, ni)
                # Skip if hybrid scorer says so
                if spi <= sni:
                    continue
                # If conditions are met, hash and enroll triple
                if self.user_as_items:
                    triple = (profile, pi, ni)
                else:
                    triple = (ui, pi, ni)
                if not hashes_container.enroll(pre_hash(triple, contains_iter=self.user_as_items)):
                    continue
                # If not seen, store sample
                samples.append((profile, pi, ni, ui))
                n -= 1
        return samples

    def strategy_2(self, samples_per_item, hashes_container):
        # Initialization
        samples = []
        if not self.user_as_items:
            assert samples_per_item == 0, "Trying to use fake strategy when real users are required"
        for pi, _ in enumerate(tqdm(self.artist_by_idx, desc="Strategy 2")):
            profile = (pi,)
            n = samples_per_item
            while n > 0:
                # Sample negative item
                while True:
                    ni = self.__sample_artwork_index(pi)
                    if ni != pi:
                        break
                # Compare visual similarity
                if self.vissimhandler.same(pi, ni):
                    continue
                # If conditions are met, hash and enroll triple
                triple = (profile, pi, ni)
                if not hashes_container.enroll(pre_hash(triple)):
                    continue
                # If not seen, store sample
                samples.append((*triple, -1))
                n -= 1
        return samples

    def strategy_3(self, n_samples_per_user, hashes_container):
        # Initialization
        interactions = self.interactions.copy()
        samples = []
        for ui, group in tqdm(interactions.groupby("user_id"), desc="Strategy 3"):
            full_profile = np.hstack(group["item_id"].values).tolist()
            artists_list = self.artist_by_idx[full_profile]
            clusters_list = self.cluster_by_idx[full_profile]
            user_margin = self.confidence_margin / len(full_profile)
            n = n_samples_per_user
            while n > 0:
                # Get profile
                if self.max_profile_size:
                    # Use the latest items only
                    profile = full_profile[-self.max_profile_size:]
                else:
                    profile = list(full_profile)
                # Sample positive and negative items
                pi = self.__sample_artwork_index_smart(artists_list, clusters_list, set(profile))
                ni = self.__sample_artwork_index_smart(artists_list, clusters_list, set(profile))
                # Skip if sample items are the same
                if pi == ni:
                    continue
                # Compare visual similarity
                if self.vissimhandler.same(pi, ni):
                    continue
                # Get score from hybrid scorer and sort accordingly
                spi = self.hybrid_scorer.get_score(ui, profile, pi)
                sni = self.hybrid_scorer.get_score(ui, profile, ni)
                if spi < sni:
                    spi, sni = sni, spi
                    pi, ni = ni, pi
                # Skip if margin is not met
                if spi < sni + user_margin:
                    continue
                # If conditions are met, hash and enroll triple
                if self.user_as_items:
                    triple = (profile, pi, ni)
                else:
                    triple = (ui, pi, ni)
                if not hashes_container.enroll(pre_hash(triple, contains_iter=self.user_as_items)):
                    continue
                # If not seen, store sample
                samples.append((profile, pi, ni, ui))
                n -= 1
        return samples

    def strategy_4(self, samples_per_item, hashes_container):
        # Initialization
        samples = []
        if not self.user_as_items:
            assert samples_per_item == 0, "Trying to use fake strategy when real users are required"
        for profile_item, _ in enumerate(tqdm(self.artist_by_idx, desc="Strategy 4")):
            profile = (profile_item,)
            n = samples_per_item
            while n > 0:
                # Sample positive item
                while True:
                    pi = self.__sample_artwork_index(profile_item)
                    if pi != profile_item:
                        break
                # Sample negative item
                while True:
                    ni = self.__sample_artwork_index(profile_item)
                    if ni != profile_item:
                        break
                # Skip if sample items are the same
                if pi == ni:
                    continue
                # Compare visual similarity
                if self.vissimhandler.same(pi, ni):
                    continue
                # Get score from hybrid scorer and sort accordingly
                spi = self.hybrid_scorer.simfunc(profile_item, pi)
                sni = self.hybrid_scorer.simfunc(profile_item, ni)
                if spi < sni:
                    spi, sni = sni, spi
                    pi, ni = ni, pi
                # Skip if margin is not met
                if spi < sni + self.confidence_margin:
                    continue
                # If conditions are met, hash and enroll triple
                triple = (profile, pi, ni)
                if not hashes_container.enroll(pre_hash(triple)):
                    continue
                # If not seen, store sample
                samples.append((*triple, -1))
                n -= 1
        return samples

    def naive_strategy_1(self, samples_per_user, hashes_container):
        # Initialization
        interactions = self.interactions.copy()
        samples = []
        for ui, group in tqdm(interactions.groupby("user_id"), desc="Naive strategy 1"):
            # Get profile artworks
            full_profile = np.hstack(group["item_id"].values).tolist()
            full_profile_set = set(full_profile)
            n = samples_per_user
            while n > 0:
                # Sample positive and negative items
                pi_index = random.randrange(len(full_profile))
                pi = full_profile[pi_index]
                # Get profile
                if self.max_profile_size:
                    # "pi_index + 1" to include pi in profile
                    profile = full_profile[max(0, pi_index - self.max_profile_size + 1):pi_index + 1]
                else:
                    profile = list(full_profile)
                # (While loop is in the sampling method)
                ni = self.__sample_artwork_index_naive(full_profile_set)

                # If conditions are met, hash and enroll triple
                if self.user_as_items:
                    triple = (profile, pi, ni)
                else:
                    triple = (ui, pi, ni)
                if not hashes_container.enroll(pre_hash(triple, contains_iter=self.user_as_items)):
                    continue
                # If not seen, store sample
                samples.append((profile, pi, ni, ui))
                n -= 1
        return samples
