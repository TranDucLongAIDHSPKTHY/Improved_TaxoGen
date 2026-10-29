"""The models compared in the experiments.

Taxonomy construction (each returns a taxorec.taxonomy.Taxonomy):

    taxogen_improved   TaxoGen-Improved, the proposed method (TaxonomyBuilder);
                       the same builder also gives the sbert_kmeans and hclus
                       baselines and every ablation variant
    taxogen            TaxoGen (Zhang et al., KDD 2018) run by its original code
    hlda               hierarchical LDA
    bertopic_flat      flat BERTopic topics

Recommendation:

    recommenders       Popularity, ItemKNN, RP3beta, BPR-MF, ContentKNN (Sem),
                       TaxonomyKNN (Tax) and Hybrid fusions

Which configuration of each model is run is set in configs/ (taxonomy.methods,
rec.models); the shared parts (encoders, clustering, CaseOLAP, relation
scoring) are in taxorec.taxonomy.components.
"""
