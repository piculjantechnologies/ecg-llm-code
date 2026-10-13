"""PTB-XL's 44 diagnostic statements and their superclasses (scp_statements.csv, diagnostic == 1)."""

STATEMENT_CLASSES = {
    "2AVB": "CD",
    "WPW": "CD",
    "CLBBB": "CD",
    "CRBBB": "CD",
    "ILBBB": "CD",
    "IVCD": "CD",
    "1AVB": "CD",
    "LAFB": "CD",
    "IRBBB": "CD",
    "3AVB": "CD",
    "LPFB": "CD",
    "SEHYP": "HYP",
    "LAO/LAE": "HYP",
    "RAO/RAE": "HYP",
    "RVH": "HYP",
    "LVH": "HYP",
    "IPLMI": "MI",
    "INJAL": "MI",
    "IPMI": "MI",
    "INJIN": "MI",
    "INJLA": "MI",
    "PMI": "MI",
    "INJAS": "MI",
    "INJIL": "MI",
    "ALMI": "MI",
    "AMI": "MI",
    "IMI": "MI",
    "ILMI": "MI",
    "LMI": "MI",
    "ASMI": "MI",
    "NORM": "NORM",
    "ISCIL": "STTC",
    "NST_": "STTC",
    "DIG": "STTC",
    "LNGQT": "STTC",
    "ISCAL": "STTC",
    "ISC_": "STTC",
    "EL": "STTC",
    "ANEUR": "STTC",
    "ISCLA": "STTC",
    "ISCIN": "STTC",
    "ISCAS": "STTC",
    "ISCAN": "STTC",
    "NDT": "STTC",
}


def aggregate(statement_probs, how="max"):
    """Superclass probabilities from statement probabilities (missing statements count as 0)."""
    out = {}
    for cls in ("NORM", "MI", "STTC", "CD", "HYP"):
        ps = [min(1.0, max(0.0, float(statement_probs.get(code, 0.0))))
              for code, c in STATEMENT_CLASSES.items() if c == cls]
        if how == "max":
            out[cls] = max(ps)
        else:
            prod = 1.0
            for p in ps:
                prod *= 1.0 - p
            out[cls] = 1.0 - prod
    return out
