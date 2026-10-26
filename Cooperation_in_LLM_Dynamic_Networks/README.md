
This collection contains experimental programs, directed-decision data, and statistical code for studying how network dynamics and reputation information shape cooperation among LLM agents. The four models are Claude Sonnet 4.5 (`ClaudeSonnet45`), DeepSeek R1 (`DSR1`), Gemini 3 Flash (`Gemini3flash`), and GPT-5.2 (`GPT52`).

## Experiments and recorded decisions

Experiments involve 20 agents interacting for 15 rounds in fully static, mixed, or fully dynamic networks. The initial dynamic-tie fraction, `Alpha`, is 0 in fully static networks, 0.1–0.9 in mixed networks, and 1 in fully dynamic networks. Mixed and fully dynamic networks have conditions with and without reputation information. Each treatment has 10 independent replications, giving 210 replications per LLM.

`Code/LLM_experiments/` contains a folder for each LLM, with separate programs for static, dynamic, dynamic-with-reputation, mixed, and mixed-with-reputation treatments. These programs define the prompts, cooperation decisions, feedback, and tie updating. Dynamic ties can be removed after rounds 3, 6, 9, and 12; static ties remain fixed, retained dynamic ties remain dynamic, and newly formed ties are dynamic.

`Data/LLM_experiments/` contains one `panel.csv` per LLM. Each row records one agent's decision toward one neighbor in one round. The combination of `Network_ID`, `Round`, `Agent`, and `Neighbor` identifies a row. Cooperation is choosing Give; defection is choosing Keep.

## Analytical sequence

The analyses first compare cooperation across network treatments and tie types at `Alpha = 0.5`. `pairwise_comparisons.py` compares replication-level cooperation rates averaged over rounds 13–15, with standard errors clustered by network replication. The reputation component of `mechanism_models.R` examines network, tie-type, and reputation effects using decisions from rounds 13–15, controlling for round and current neighbor count.

`alpha_models.R` extends the comparisons across initial dynamic-tie fractions, examining mixed networks overall, mixed static ties, and mixed dynamic ties. Alpha is categorical. Model 1 combines reputation conditions; Model 2 includes reputation and its interactions with alpha. Reported network comparisons use fully static networks as the reference.

The remaining components of `mechanism_models.R` examine static-tie cooperation following tie-dropping opportunities (`T`, rounds 2–15) and following loss of an existing tie (`A`, rounds 4, 7, 10, and 13). These models control for reputation, round, current neighbor count, and previous-round neighborhood cooperation. The R models include random intercepts for network replications and agents within replications. Pairwise differences are in percentage points; mixed-model coefficients are in log-odds units.

The statistical programs are shared across LLMs. The supplied data consist of the four decision panels; the R scripts also reference prepared binomial or event-history inputs and optimizer starting values, which are separate from the retained files.

## Reading the data fields

| Fields | Meaning |
|---|---|
| `NetworkType`, `NetType`, `Alpha`, `Reputation` | Network-treatment labels, initial dynamic-tie fraction, and reputation-information condition (`1` with information; `0` without). |
| `Repeat`, `Network_ID` | Replication number within a treatment and the unique network-replication identifier within an LLM. |
| `Round`, `Agent`, `Neighbor` | Experimental round, acting agent, and recipient of the directed decision. |
| `Action`, `Action_Num` | Give/Keep decision and its binary coding: Give = 1, Keep = 0. |
| `Participant_ID`, `Round_ID` | Identifiers for an agent within a network replication and for an agent-round observation. |
| `TieType`, `DynamicTie`, `LegacyTieType` | Current static/dynamic tie type, its binary indicator (`1` dynamic), and the original exported tie label. `TieType` and `DynamicTie` contain the corrected analysis classifications. |
| `Network_Size`, `Snapshot_Nodes`, `Number_of_Ties` | Initial agent count, node count in the pre-decision network snapshot, and the acting agent's current neighbor count. |
| `Reputation_Score` | The acting agent's cumulative percentage of Give decisions before the current round; initialized to 100 when there is no prior history. |
| `Prev_Alter_Action`, `PropAltersPrev` | The recipient's previous-round decision toward the actor and its mean over the actor's current neighbors with observed previous-round decisions. |
| `PostDrop`, `Dropped_Previous` | Indicator for rounds immediately after tie-dropping opportunities (4, 7, 10, 13), and a legacy tie-loss indicator comparing the snapshots preceding rounds t−2 and t−1. |
| `DynamicNetwork`, `OverallCategory`, `TieCategory` | Fully dynamic-network indicator and categorical labels for overall-network and network/tie comparisons. |
| `Action_Source` | Outcome provenance: `original_export`, `uniquely_recovered`, or `unresolved`. |

Blank action values indicate unresolved outcomes in early rounds. All action outcomes in rounds 7–15 are available. Cooperation rates are calculated from directed decisions within each network-round, with rounds and replications weighted equally for replication-level comparisons.
