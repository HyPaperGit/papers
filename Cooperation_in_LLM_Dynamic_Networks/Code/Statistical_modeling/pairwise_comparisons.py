from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
DATA = args.data
OUT = args.output


MODELS = {
    "ClaudeSonnet45": "Claude Sonnet 4.5",
    "DSR1": "DeepSeek R1",
    "Gemini3flash": "Gemini 3 Flash",
    "GPT52": "GPT-5.2",
}

CATEGORIES = [
    "Fully dynamic",
    "Mixed",
    "Fully static",
    "Mixed: dynamic ties",
    "Mixed: static ties",
]

PAIRS = [
    ("Fully dynamic", "Fully static"),
    ("Fully dynamic", "Mixed"),
    ("Fully dynamic", "Mixed: static ties"),
    ("Fully dynamic", "Mixed: dynamic ties"),
    ("Mixed", "Fully static"),
    ("Mixed", "Mixed: static ties"),
    ("Mixed", "Mixed: dynamic ties"),
    ("Mixed: static ties", "Fully static"),
    ("Mixed: dynamic ties", "Fully static"),
    ("Mixed: dynamic ties", "Mixed: static ties"),
]


def build_rates() -> pd.DataFrame:
    rows = []
    for slug, llm in MODELS.items():
        d = pd.read_csv(DATA / slug / "panel.csv", low_memory=False)
        d["Trials"] = 1
        d = d[d["Alpha"].isin([0, 0.5, 1]) & d["Round"].isin([13, 14, 15])].copy()
        assert d["Network_ID"].nunique() == 50
        assert set(d["Round"]) == {13, 14, 15}

        specs = [
            ("Fully static", d["NetType"].eq("Static")),
            ("Fully dynamic", d["NetType"].eq("Dynamic")),
            ("Mixed", d["NetType"].eq("Mixed")),
            ("Mixed: static ties", d["NetType"].eq("Mixed") & d["TieType"].eq("Static")),
            ("Mixed: dynamic ties", d["NetType"].eq("Mixed") & d["TieType"].eq("Dynamic")),
        ]

        for category, mask in specs:
            q = d[mask]
            by_round = (
                q.groupby(["Network_ID", "Round"], as_index=False)
                .agg(Cooperation=("Action_Num", "sum"), Actions=("Trials", "sum"))
            )
            assert (by_round["Actions"] > 0).all()
            by_round["Rate"] = 100 * by_round["Cooperation"] / by_round["Actions"]
            by_run = by_round.groupby("Network_ID", as_index=False).agg(
                Rate=("Rate", "mean"), Rounds=("Round", "nunique")
            )
            assert by_run["Rounds"].eq(3).all()
            expected = 10 if category == "Fully static" else 20
            assert len(by_run) == expected
            by_run["LLM"] = llm
            by_run["Category"] = category
            rows.append(by_run[["LLM", "Network_ID", "Category", "Rate"]])

    z = pd.concat(rows, ignore_index=True)
    z["Cluster"] = z["LLM"] + "|" + z["Network_ID"].astype(str)
    assert z["Cluster"].nunique() == 200
    assert len(z) == 360

    return z


def fit_and_contrast(z: pd.DataFrame) -> pd.DataFrame:
    cells = [f"{llm}|{cat}" for llm in MODELS.values() for cat in CATEGORIES]
    observed = z["LLM"] + "|" + z["Category"]
    x = pd.DataFrame({cell: observed.eq(cell).astype(float) for cell in cells})
    fit = sm.OLS(z["Rate"].to_numpy(), x.to_numpy()).fit()
    robust = fit.get_robustcov_results(
        cov_type="cluster",
        groups=z["Cluster"].to_numpy(),
        use_correction=True,
        df_correction=True,
        use_t=True,
    )
    beta = np.asarray(robust.params)
    vcov = np.asarray(robust.cov_params())
    df = z["Cluster"].nunique() - 1
    crit = student_t.ppf(0.975, df)
    index = {cell: i for i, cell in enumerate(cells)}

    def vector(llms, first, second):
        v = np.zeros(len(cells))
        weight = 1 / len(llms)
        for llm in llms:
            v[index[f"{llm}|{first}"]] += weight
            v[index[f"{llm}|{second}"]] -= weight
        return v

    out = []
    blocks = [("---", list(MODELS.values()))] + [(llm, [llm]) for llm in MODELS.values()]
    for block, llms in blocks:
        for first, second in PAIRS:
            v = vector(llms, first, second)
            estimate = float(v @ beta)
            se = float(np.sqrt(v @ vcov @ v))
            statistic = estimate / se
            p = float(2 * student_t.sf(abs(statistic), df))
            out.append(
                {
                    "LLM": block,
                    "First": first,
                    "Second": second,
                    "Difference": estimate,
                    "SE_CR1": se,
                    "LCB": estimate - crit * se,
                    "UCB": estimate + crit * se,
                    "t": statistic,
                    "df": df,
                    "P": p,
                }
            )
    result = pd.DataFrame(out)
    assert len(result) == 50
    return result


if __name__ == "__main__":
    result = fit_and_contrast(build_rates())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))
