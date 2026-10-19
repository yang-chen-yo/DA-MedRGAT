def visit_recency_weights(triples, half_life=3.0):
    """Recency weight w = 0.5 ** (k / half_life) for each triple of one patient.

    k is the rank of the triple's visit among the patient's visits, ordered
    from most recent (k = 0) to oldest. All triples from the same visit share
    the same weight. `triples` must all belong to a single patient.
    """
    visit_dates = {t["visit_id"]: t["visit_date"] for t in triples}
    order = sorted(visit_dates, key=visit_dates.get, reverse=True)
    rank = {vid: k for k, vid in enumerate(order)}
    return [0.5 ** (rank[t["visit_id"]] / half_life) for t in triples]
