

"""
Mixed-network LLM-driven simulation (Parallel Version)
"""

import csv
import time
import string
import numpy as np
import networkx as nx
from openai import OpenAI
from collections import deque
from typing import List, Dict, Tuple, Optional
import os
import concurrent.futures
import threading


print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def generate_agent_ids(N: int) -> List[str]:
    ids = []
    letters = string.ascii_uppercase
    for v in letters:
        for j in letters:
            if v != j:
                ids.append(v + j)
                if len(ids) == N:
                    return ids
    raise ValueError("N too large for current ID generator")


class MixedLLMCooperationSim:
    def __init__(
            self,
            N: int,
            L: int,
            m: int,
            p: float,
            l: int = 3,
            api_key: str = "",
            alpha: float = 0.5,
            give_cost: int = 50,
            give_gain: int = 100,
            api_interval: float = 1.0,
            initial_points: int = 1000,
            models: str = "gemini-3-flash-preview",
            base_url: str = "https://www.dmxapi.cn/v1",
            run_id: int = 0
    ):
        self.N, self.p, self.L, self.m, self.alpha, self.l_dyn = N, p, L, m, alpha, l
        self.model = models
        self.api_key = api_key
        self.base_url = base_url
        self.give_gain = give_gain
        self.give_cost = give_cost
        self.decision_networks = []
        self.api_interval = api_interval
        self.initial_points = initial_points
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.run_id = run_id


        self.agent_ids = generate_agent_ids(N)


        max_edges = N * (N - 1) // 2
        M = round(p * max_edges)
        self.G = nx.gnm_random_graph(N, M)
        self.G = nx.relabel_nodes(self.G, dict(zip(range(N), self.agent_ids)))


        all_edges = list(self.G.edges())
        np.random.shuffle(all_edges)

        M_dynamic = round(M * self.alpha)

        for i_num, (u, v) in enumerate(all_edges):
            if i_num < M_dynamic:
                self.G[u][v]['dynamic'] = True
            else:
                self.G[u][v]['dynamic'] = False

        self.networks_history: List[nx.Graph] = [self.G.copy()]


        self.memory = [deque(maxlen=m) for _ in range(N)]
        self.points = np.full(N, initial_points, dtype=int)
        self.payoff_matrix = np.zeros((N, L), dtype=int)
        self.reputation_matrix = np.zeros((N, L), dtype=float)

        self.decision_matrix = [[{} for _ in range(L)] for _ in range(N)]


        self._id2idx = {aids: v for v, aids in enumerate(self.agent_ids)}


    def _system_prompt(self) -> str:
        return (
            "Welcome in today’s study, you will be making decisions with other participants.\n"
            "Both your own decisions and the decisions of the others will influence your earnings. "
            "Therefore, it is very important that you read all instructions carefully.\n"
            "During the study, you will be connected to some of the other participants. "
            "Each participant will be identified only with a letter ID (e.g., Participant X or Participant Y).\n"
            "You and the others you are connected to will make decisions in several different phases. "
            "Each phase consists of multiple rounds.\n"
            "These decisions will cause you and the others to gain or lose points. "
            f"You start with {self.initial_points} points.\n"
            "At the end of the study, you will get a final number of points you earned in one randomly selected phase "
            "of the study."
            "So, each phase is important.\n"
            "Specifically: in each round, you will choose whether to give points to each of the participant you are "
            "connected to."
            "All others will also choose whether to give points to each of the participant they are connected to.\n"
            "For each participant you are connected to has the same choice with each of the others they are connected "
            "to, including you."
            f"If the other chooses Give, the other pays {self.give_cost} points for you to gain {self.give_gain} points. "
            f"If the other chooses Keep, you don’t gain any points.\n"
            "Once everyone makes their decisions, the results are displayed. "
            "You will be shown the choices of each other you are connected to, and how many points in total you "
            "gained or lost."
        )

    def _phase_prompt(self) -> str:
        return (
            "In this phase, periodically (that is, in some rounds), after you learn the results, "
            "you will be asked whether or not you want to stop interacting with one of the others you are connected to,"
            " and connect with a new participant.\n"
            "If you do, you will choose which one other you no longer wish to interact with. "
            "However, you can only disconnect from some of the others you are connected to. "
            "This indicates that, in some rounds, you will be able to disconnect from this other. "
            "You will see an ID list of participants allowed to disconnect, while the remaining participants, if any, "
            "that you will be connected to throughout the remainder of this phase.\n"
            "While you are deciding whether and with whom you wish to stop interacting, "
            "the others will be deciding whether they wish to continue interacting with their ties, including you.\n"
            "In those rounds that you are able to, if you choose to stop interacting with one of the others, "
            "you will choose which one new other you want to connect with.\n"
            f"Specifically, you will see a list containing the Participant IDs of all of the new others with whom you "
            f"could potentially connect. You will also see the percentage of the them they chose Give during this "
            f"phase of the study—that is, the percentage of the time they chose to pay {self.give_cost} points to "
            f"give another {self.give_gain} points during this phase. From the list, you can select one new other to "
            f"connect to.\n"
            "You will have the opportunity to connect with a new player even if you previously "
            "disconnected from that same player.\n"
            "The other must confirm this tie in order to finalize it. Thus, the other will see the percentage of the "
            "time you chose Give during this phase of the study. Then the other will be asked if he or she would like "
            "to connect with you. If they do, you will successfully be connected with the new other.\n"
            "Similarly, you will be given an opportunity to confirm any ties others want to form with you. You will "
            "see the percentage of time they chose Give during this phase of the study.\n"
            "Importantly: remember that in this phase you can only disconnect from some of the others you are "
            "connected to. However, when choosing new ties, you will see the percentage of the time the others chose "
            "Give toward all of the others they are connected to (both those they can stop connecting with, "
            "and those they stay connected to throughout the phase). Likewise, the others will see the percentage of "
            "the time you chose Give toward all of the others you are connected to.\n"
            "Be sure you have carefully read the instructions."
        )

    def _decision_prompt_base(self, agent_id: str, current_points: int, round_num: int = 0) -> str:
        neighbors = list(self.G.neighbors(agent_id))
        neighbor_str = ", ".join(neighbors) if neighbors else "None"

        base_prompt = (
            f"Your ID: {agent_id}, Your point: {current_points}\n"
            f"Neighbor ID list: {neighbor_str}\n"
            "Please choose to Give or Keep:\n"
            f"\tIf you choose Give you will pay {self.give_cost} points to give your neighbor {self.give_gain} points\n"
            f"\tIf you choose Keep you will pay 0 points to give your neighbor 0 points\n"
            "Do you want to choose Give or Keep with this neighbor?\n\n"
            "**Response Format**\n"
            "```plaintext\n"
            "For neighbor {ID}, I choose {ACTION}\n"
            "```"
        )

        if round_num == 0:
            random_prompt = (
                "\nThis is the first round. To ensure diverse initial strategies. "
                "Please mix Give and Keep choices among your neighbors. "
                "Randomly select some neighbors to Give and others to Keep."
            )
            base_prompt += random_prompt

        return base_prompt

    def _results_prompt(self, agent_id: str, decisions: Dict[str, str], received: Dict[str, str]) -> str:
        give_dec = [n for n, d in decisions.items() if d == "Give"]
        keep_dec = [n for n, d in decisions.items() if d == "Keep"]
        give_rec = [n for n, d in received.items() if d == "Give"]
        keep_rec = [n for n, d in received.items() if d == "Keep"]
        return (
            "Results\n"
            f"You chose Give with {len(give_dec)} player(s)—{','.join(give_dec)}—"
            f"and paid a total of {self.give_cost * len(give_dec)} points to give {self.give_gain} points to each of "
            f"them.\n"
            f"You chose Keep with {len(keep_dec)} player(s)—{','.join(keep_dec)}—"
            f"and paid 0 points to give 0 points to each of them.\n"
            f"{len(give_rec)} player(s) you are connected with chose Give—{','.join(give_rec)}—"
            f"and gave you a total of {self.give_gain * len(give_rec)} points.\n"
            f"{len(keep_rec)} player(s) you are connected with chose Keep—{','.join(keep_rec)}—"
            f"and gave you 0 points."
        )


    def _parse_decisions(self, text: str, neighbors: List[str]) -> Dict[str, str]:
        out = {n: "Keep" for n in neighbors}
        for line in text.splitlines():
            if "For neighbor" in line and "I choose" in line:
                try:
                    parts = line.strip().split()
                    nid = parts[2].rstrip(",")
                    action = parts[-1]
                    if nid in out and action in {"Give", "Keep"}:
                        out[nid] = action
                except Exception:
                    continue
        return out

    def _call_llm(self, messages: List[dict]) -> str:
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,

                    stream=False,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:

                time.sleep(min(30, 2 ** attempt))
        return ""


    def _calculate_reputation(self, agent_idx: int, current_round: int) -> float:
        """Calculate the agent's reputation before the specified round (percentage of Give actions)."""
        if current_round < 0:
            return 0.0

        total_actions = 0
        give_actions = 0


        for rounds in range(current_round + 1):
            decisions = self.decision_matrix[agent_idx][rounds]
            total_actions += len(decisions)
            give_actions += sum(1 for action in decisions.values() if action == "Give")

        return (give_actions / total_actions * 100) if total_actions > 0 else 0.0


    def _process_single_decision(self, v: int, aids: str, rounds: int, system: str, phase: str):
        """Process decisions in parallel."""

        if aids not in self.G or self.G.degree(aids) == 0:
            return aids, {}, None

        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[v]:
            msgs.append({"role": "user", "content": mem["phase"]})
            msgs.append({"role": "user", "content": mem["decision"]})
            msgs.append({"role": "assistant", "content": mem["decision_resp"]})
            msgs.append({"role": "user", "content": mem["results"]})

        decision_prompt = self._decision_prompt_base(aids, self.points[v], rounds)
        msgs.append({"role": "user", "content": phase})
        msgs.append({"role": "user", "content": decision_prompt})

        resp = self._call_llm(msgs)
        neighbors = list(self.G.neighbors(aids))
        decisions = self._parse_decisions(resp, neighbors)

        mem_update = {
            "phase": phase,
            "decision": decision_prompt,
            "decision_resp": "\n".join([f"For neighbor {n}, I choose {d}" for n, d in decisions.items()]),
            "results": "",
        }
        return aids, decisions, mem_update

    def _process_single_cut(self, v: int, aids: str, decisions_all: Dict, system: str, phase: str):
        """Process tie removal in parallel; only dynamic ties can be removed in mixed networks."""
        if aids not in self.G: return aids, None

        neighbors = list(self.G.neighbors(aids))
        if not neighbors: return aids, None


        dynamic_neighbors = [n for n in neighbors if self.G[aids][n].get('dynamic', False)]
        if not dynamic_neighbors: return aids, None

        give_by_nbs = [n for n in neighbors if decisions_all.get(n, {}).get(aids, "Keep") == "Give"]
        keep_by_nbs = [n for n in neighbors if decisions_all.get(n, {}).get(aids, "Keep") == "Keep"]
        cut_list_str = ", ".join(dynamic_neighbors)

        cut_prompt = (
            "You may choose to cut your ties with one player.\n"
            f"Players {','.join(give_by_nbs)} chose to Give, and Players {','.join(keep_by_nbs)} chose to Keep to "
            f"you this round.\n"
            f"Tie-cutting ID list: {cut_list_str}\n"
            'Either select a player from tie-cutting list to cut your ties with or select "No player".'
        )

        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[v]:
            msgs.append({"role": "user", "content": mem["phase"]})
            msgs.append({"role": "user", "content": mem["decision"]})
            msgs.append({"role": "assistant", "content": mem["decision_resp"]})
            msgs.append({"role": "user", "content": mem["results"]})
        msgs.append({"role": "user", "content": phase})
        msgs.append({"role": "user", "content": cut_prompt})

        resp = self._call_llm(msgs).strip()
        chosen = None
        for line in resp.splitlines():
            if "No player" in line or "no player" in line.lower():
                break
            for nb in dynamic_neighbors:
                if nb in line:
                    chosen = nb
                    break
            if chosen: break
        return aids, chosen

    def _process_single_make(self, v: int, req: str, system: str, phase: str, candidates: List[str],
                             current_round: int):
        """Process tie-formation requests in parallel with reputation information."""
        if req not in self.G: return req, None

        current_nbs = set(self.G.neighbors(req))
        possible = [t for t in candidates if t != req and t not in current_nbs]
        if not possible: return req, None


        poss_with_reputation = []
        for cand in possible:
            cand_idx = self._id2idx[cand]
            rep_value = self._calculate_reputation(cand_idx, current_round)
            rep_str = f"{rep_value:.2f}%"
            poss_with_reputation.append(f"{cand}({rep_str})")
        poss_str = ", ".join(poss_with_reputation)

        make_prompt = (
            "Make a new tie.\n"
            f"Reputation indicates the percentage of times each player chose 'Give' in previous "
            "rounds (including current round). Reputation range: [0%-100%].\n"
            "Please choose one player from tie-making list with whom to make a new tie, taking their reputation "
            "into consideration.\n"
            f"Tie-making ID list with reputations: {poss_str}\n"
            "You will have the opportunity to connect with a new player even if you previously "
            "disconnected from that same player.\n"
            "If they agree, you will participate in future rounds with this player."
        )

        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[v]:
            msgs.append({"role": "user", "content": mem["phase"]})
            msgs.append({"role": "user", "content": mem["decision"]})
            msgs.append({"role": "assistant", "content": mem["decision_resp"]})
            msgs.append({"role": "user", "content": mem["results"]})
        msgs.append({"role": "user", "content": phase})
        msgs.append({"role": "user", "content": make_prompt})

        resp = self._call_llm(msgs).strip()
        chosen = None
        for line in resp.splitlines():
            for cand in possible:
                if cand in line:
                    chosen = cand
                    break
        return req, chosen

    def _process_single_accept(self, aa: str, bb: str, system: str, phase: str, current_round: int):
        """Process tie-formation acceptance in parallel with reputation information."""
        if bb not in self.G: return aa, bb, False

        bb_idx = self._id2idx[bb]
        aa_idx = self._id2idx[aa]

        aa_rep_value = self._calculate_reputation(aa_idx, current_round)
        aa_rep_str = f"{aa_rep_value:.2f}%"

        consent_prompt = (
            f"Participant {aa} (reputation: {aa_rep_str}%) "
            f"requests to form a new tie with you. Do you accept? "
            f"Reputation indicates the percentage of times this player chose 'Give' in previous rounds. "
            f"Reputation range: [0%-100%]\n"
            f"Please answer exactly: Accept or Reject."
        )

        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[bb_idx]:
            msgs.append({"role": "user", "content": mem["phase"]})
            msgs.append({"role": "user", "content": mem["decision"]})
            msgs.append({"role": "assistant", "content": mem["decision_resp"]})
            msgs.append({"role": "user", "content": mem["results"]})
        msgs.append({"role": "user", "content": phase})
        msgs.append({"role": "user", "content": consent_prompt})

        resp = self._call_llm(msgs).strip()
        return aa, bb, ("accept" in resp.lower() or "Accept" in resp)


    def _run_single_round(self, rounds: int):

        self.decision_networks.append(self.G.copy())

        system = self._system_prompt()
        phase = self._phase_prompt()


        decisions_all: Dict[str, Dict[str, str]] = {}
        memory_temp = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N) as executor:
            futures = {
                executor.submit(self._process_single_decision, v, aids, rounds, system, phase): aids
                for v, aids in enumerate(self.agent_ids)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    res_aid, res_dec, res_mem = future.result()
                    decisions_all[res_aid] = res_dec
                    if res_mem:
                        memory_temp[res_aid] = res_mem
                except Exception:
                    pass


        for v, aids in enumerate(self.agent_ids):

            self.decision_matrix[v][rounds] = decisions_all.get(aids, {})


            if aids in memory_temp:
                self.memory[v].append(memory_temp[aids])
            elif aids in self.G:
                self.memory[v].append({"phase": phase, "decision": "", "decision_resp": "", "results": ""})


        for v in range(self.N):
            self.reputation_matrix[v, rounds] = self._calculate_reputation(v, rounds)


        for v, aids in enumerate(self.agent_ids):
            if aids not in self.G or self.G.degree(aids) == 0:
                self.payoff_matrix[v, rounds] = 0
                continue

            neighbors = list(self.G.neighbors(aids))
            my_dec = decisions_all.get(aids, {})
            cost = sum(self.give_cost for d in my_dec.values() if d == "Give")

            rec = {}
            for nb in neighbors:
                rec[nb] = decisions_all.get(nb, {}).get(aids, "Keep")

            gain = sum(self.give_gain for d in rec.values() if d == "Give")
            payoff = gain - cost
            self.payoff_matrix[v, rounds] = payoff
            self.points[v] += payoff

            res_prompt = self._results_prompt(aids, my_dec, rec)
            if self.memory[v]:
                self.memory[v][-1]["results"] = res_prompt


        if (rounds + 1) % self.l_dyn == 0:
            self._rewiring_phase(decisions_all, rounds, system, phase)


        self.networks_history.append(self.G.copy())

    def _rewiring_phase(self, decisions_all: Dict, current_round: int, system: str, phase: str):

        to_cut: Dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N) as executor:
            futures = {
                executor.submit(self._process_single_cut, v, aids, decisions_all, system, phase): aids
                for v, aids in enumerate(self.agent_ids)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    res_aid, res_target = future.result()
                    if res_target:
                        to_cut[res_aid] = res_target
                except Exception:
                    pass


        for aa, bb in to_cut.items():
            if self.G.has_edge(aa, bb):
                self.G.remove_edge(aa, bb)


        requesters = list(to_cut.keys())
        candidates = [aids for aids in self.agent_ids if aids in self.G and self.G.degree(aids) > 0]
        new_edges = []

        if requesters:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(requesters)) as executor:
                futures = {
                    executor.submit(self._process_single_make, self._id2idx[req], req, system, phase, candidates,
                                    current_round): req
                    for req in requesters
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res_req, res_target = future.result()
                        if res_target:
                            new_edges.append((res_req, res_target))
                    except Exception:
                        pass


        seen_pairs = set()
        accept_tasks = []
        for aa, bb in new_edges:
            if (aa, bb) in seen_pairs or (bb, aa) in seen_pairs: continue
            seen_pairs.add((aa, bb))
            accept_tasks.append((aa, bb))

        if accept_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(accept_tasks)) as executor:
                futures = {
                    executor.submit(self._process_single_accept, aa, bb, system, phase, current_round): (aa, bb)
                    for aa, bb in accept_tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res_aa, res_bb, accepted = future.result()
                        if accepted:

                            if res_aa in self.G and res_bb in self.G and not self.G.has_edge(res_aa,
                                                                                             res_bb) and res_aa != res_bb:
                                self.G.add_edge(res_aa, res_bb)
                                self.G[res_aa][res_bb]['dynamic'] = True
                    except Exception:
                        pass


    def run_simulation(self):
        for rounds in range(self.L):
            safe_print(f"[Run {self.run_id}] --- Round {rounds + 1}/{self.L}  (edges: {self.G.number_of_edges()}) ---")
            self._run_single_round(rounds)
        return self._package_results()

    def _package_results(self):

        give_ratio = np.zeros((self.L, 2))
        give_ratio_by_type = np.zeros((self.L, 3))

        for rounds in range(self.L):
            decision_net = self.decision_networks[rounds]
            give_cnt = total_cnt = 0
            static_give = static_total = dynamic_give = dynamic_total = 0

            for v, aids in enumerate(self.agent_ids):
                if aids not in decision_net: continue
                neighbors = list(decision_net.neighbors(aids))

                for nb in neighbors:
                    if nb not in self.decision_matrix[v][rounds]: continue
                    act = self.decision_matrix[v][rounds][nb]

                    total_cnt += 1
                    if act == "Give": give_cnt += 1

                    is_dynamic = decision_net[aids][nb].get('dynamic', False)
                    if is_dynamic:
                        dynamic_total += 1
                        if act == "Give": dynamic_give += 1
                    else:
                        static_total += 1
                        if act == "Give": static_give += 1

            give_ratio[rounds, 0] = rounds + 1
            give_ratio[rounds, 1] = give_cnt / total_cnt if total_cnt else 0.0

            give_ratio_by_type[rounds, 0] = rounds + 1
            give_ratio_by_type[rounds, 1] = static_give / static_total if static_total else 0.0
            give_ratio_by_type[rounds, 2] = dynamic_give / dynamic_total if dynamic_total else 0.0

        return {
            "payoff_matrix": self.payoff_matrix,
            "decision_matrix": self.decision_matrix,
            "network_snapshots": self.networks_history,
            "give_ratio_per_round": give_ratio,
            "give_ratio_by_type_per_round": give_ratio_by_type,
            "final_points": self.points.copy(),
            "agent_ids": self.agent_ids,
            "reputation_matrix": self.reputation_matrix,
        }


def run_single_experiment_set(i, alpha_list, N_num, L_num, p_num, memory_num, l_dyn_num, api_keys, model):
    safe_print(f"\n--- Starting Repeat {i}/10 ---")
    model_name_safe = model.replace("/", "_")
    dir_name = f"mixed_rep_{model_name_safe}_repeat={i}"
    os.makedirs(dir_name, exist_ok=True)
    safe_print(f"[Repeat {i}] Results will be saved in: {dir_name}")

    for a in alpha_list:
        safe_print(f"[Repeat {i}] Running alpha = {a} ...")

        sim = MixedLLMCooperationSim(
            N=N_num, p=p_num, L=L_num, m=memory_num, l=l_dyn_num, alpha=a,
            give_cost=50,
            give_gain=100,
            api_interval=0.1,
            api_key=api_keys,
            initial_points=1000,
            models=model,
            run_id=i
        )
        results = sim.run_simulation()


        for r, g in enumerate(results["network_snapshots"]):
            gexf_filename = "mixed_rep_net_r={}_N={}_L={}_p={}_m={}_l={}_a={}_mod={}.gexf".format(
                r, N_num, L_num, p_num, memory_num, l_dyn_num, a, model_name_safe
            )
            nx.write_gexf(g, os.path.join(dir_name, gexf_filename))

        csv_payoff = 'mixed_rep_pay_mat_N={}_L={}_p={}_m={}_l={}_a={}_mod={}.csv'.format(
            N_num, L_num, p_num, memory_num, l_dyn_num, a, model_name_safe
        )
        with open(os.path.join(dir_name, csv_payoff), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Round'] + results['agent_ids']
            writer.writerow(header)
            for r in range(L_num):
                writer.writerow([r + 1] + list(results['payoff_matrix'][:, r]))

        txt_dec = "mixed_rep_dec_mat_N={}_L={}_p={}_m={}_l={}_a={}_mod={}.txt".format(
            N_num, L_num, p_num, memory_num, l_dyn_num, a, model_name_safe
        )
        with open(os.path.join(dir_name, txt_dec), "w") as f:
            for i_agent, aid in enumerate(results["agent_ids"]):
                f.write(f"Agent {aid}\n")
                for r in range(L_num):
                    f.write(f"  Round {r + 1}: {results['decision_matrix'][i_agent][r]}\n")

        csv_rep = 'mixed_rep_rep_value_mat_N={}_L={}_p={}_m={}_l={}_a={}_mod={}.csv'.format(
            N_num, L_num, p_num, memory_num, l_dyn_num, a, model_name_safe
        )
        with open(os.path.join(dir_name, csv_rep), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['AgentID'] + [f'Round{r}' for r in range(L_num)]
            writer.writerow(header)
            for i_agent, aid in enumerate(results["agent_ids"]):
                row = [aid] + [f"{rep:.4f}" for rep in results["reputation_matrix"][i_agent]]
                writer.writerow(row)


        safe_print(f"--- [Repeat {i} | alpha={a}] Finished ---")


if __name__ == "__main__":

    N_num = 20
    L_num = 15
    p_num = 0.3
    l_dyn_num = 3
    memory_num = 1
    alpha_num = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    api_keys = os.environ["LLM_API_KEY"]
    model = "gemini-3-flash-preview"

    print("Starting 10 parallel mixed-reputation experiments...")

    max_parallel_experiments = 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_experiments) as executor:
        futures = []
        for i in range(1, 11):
            futures.append(
                executor.submit(run_single_experiment_set, i, alpha_num, N_num, L_num, p_num, memory_num, l_dyn_num,
                                api_keys, model)
            )
        concurrent.futures.wait(futures)

    print("\n--- All 10 repeats complete. ---")
