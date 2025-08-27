
def get_predicates(cluster:dict) -> list[str]:
    predicates = []
    for i in range(1, len(cluster)+1):
        cluster_key = f"subset{i}"
        if cluster_key in cluster:
            predicates.extend(cluster[cluster_key])
    return predicates