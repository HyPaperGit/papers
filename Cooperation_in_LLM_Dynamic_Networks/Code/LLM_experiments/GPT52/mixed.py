
"""
Mixed-network LLM-driven simulation (Parallel Version)
"""

import os
import csv
import time
import string
import numpy as np
import networkx as nx
from openai import OpenAI
from collections import deque
from typing import List, Dict, Tuple, Set, Optional
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
            models: str = "gpt-5.1",
            base_url: str = "https://www.dmxapi.cn/v1",
            run_id: int = 0
    ):
        self.model = models
        self.api_key = api_key
        self.base_url = base_url
        self.give_cost = give_cost
        self.give_gain = give_gain
        self.api_interval = api_interval
        self.initial_points = initial_points
        self.N, self.p, self.L, self.m, self.alpha, self.l_dyn = N, p, L, m, alpha, l
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
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


        self.decision_networks: List[nx.Graph] = []
        self.networks_history: List[nx.Graph] = [self.G.copy()]


        self.memory = [deque(maxlen=m) for _ in range(N)]
        self.points = np.full(N, initial_points, dtype=int)
        self.payoff_matrix = np.zeros((N, L), dtype=int)

        self.decision_matrix = [[{} for _ in range(L)] for _ in range(N)]


        self.edge_counts = []
        self.isolated_nodes = set()
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
            f"If the other chooses Give,the other pays {self.give_cost} points for you to gain {self.give_gain} points."
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
            "Specifically, you will see a list containing the Participant IDs of all of the new others with whom you "
            "could potentially connect.From the list, you can select one other to connect to.\n"
            "You will have the opportunity to connect with a new player even if you previously "
            "disconnected from that same player.\n"
            "The other must confirm this tie in order to finalize it.Thus, the other will be asked if he or she would "
            "like to connect with you.If they do, you will successfully be connected with the new other.\n"
            "Similarly, you will be given an opportunity to confirm any ties others want to form with you.\n"
            "Be sure you have carefully read the instructions."
        )

    def _decision_prompt_base(self, agent_id: str, current_points: int, rounds_num: int = 0) -> str:
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


        if rounds_num == 0:
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
            line = line.strip()
            if line.startswith("For neighbor") and "I choose" in line:
                try:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue

                    id_part = parts[0].split()[-1]
                    action = parts[1].split()[-1]
                    if id_part in out and action in {"Give", "Keep"}:
                        out[id_part] = action
                except Exception:
                    continue
        return out

    def _call_llm(self, messages: List[dict]) -> str:
        if not self.client:
            return "Give" if np.random.rand() > 0.5 else "Keep"

        for attempt in range(5):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    stream=False,
                    reasoning={"effort": "medium"},
                )
                return resp.output[1].content[0].text.strip()
            except Exception as e:

                time.sleep(min(30, 2 ** attempt))
        return ""


    def _process_single_decision(self, v: int, aids: str, r: int, system: str, phase: str):
        """Process decisions in parallel."""
        if aids in self.isolated_nodes or self.G.degree(aids) == 0:
            return aids, {}, None

        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[v]:
            msgs.append({"role": "user", "content": mem.get("phase", "")})
            msgs.append({"role": "user", "content": mem.get("decision", "")})
            msgs.append({"role": "assistant", "content": mem.get("decision_resp", "")})
            msgs.append({"role": "user", "content": mem.get("results", "")})

        decision_prompt = self._decision_prompt_base(aids, self.points[v], r)
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
        if aids in self.isolated_nodes:
            return aids, None

        neighbors = list(self.G.neighbors(aids))
        if not neighbors:
            return aids, None


        dynamic_neighbors = [n for n in neighbors if self.G[aids][n].get('dynamic', False)]
        if not dynamic_neighbors:
            return aids, None

        give_by_nbs = [n for n in neighbors if decisions_all.get(n, {}).get(aids) == "Give"]
        keep_by_nbs = [n for n in neighbors if decisions_all.get(n, {}).get(aids) == "Keep"]


        cut_prompt = (
            "You may choose to cut your ties with one player.\n"
            f"Players {','.join(give_by_nbs)} chose to Give, and Players {','.join(keep_by_nbs)} chose to Keep to "
            f"you this round.\n"
            f"Tie-cutting ID list: {', '.join(dynamic_neighbors)}\n"
            'Either select a player from tie-cutting list to cut your ties with or select "No player".'
        )

        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[v]:
            msgs.append({"role": "user", "content": mem.get("phase", "")})
            msgs.append({"role": "user", "content": mem.get("decision", "")})
            msgs.append({"role": "assistant", "content": mem.get("decision_resp", "")})
            msgs.append({"role": "user", "content": mem.get("results", "")})
        msgs.append({"role": "user", "content": phase})
        msgs.append({"role": "user", "content": cut_prompt})

        resp = self._call_llm(msgs).strip()
        chosen = None
        for line in resp.splitlines():
            line = line.strip()
            if "no player" in line.lower() or "none" in line.lower():
                break
            for nb in dynamic_neighbors:
                if nb in line:
                    chosen = nb
                    break
            if chosen:
                break

        return aids, chosen

    def _process_single_make(self, v: int, requester: str, system: str, phase: str):
        """Process tie-formation requests in parallel."""
        if requester in self.isolated_nodes or self.G.degree(requester) == 0:
            return requester, None

        current_nbs = set(self.G.neighbors(requester))

        possible_targets = [t for t in self.agent_ids
                            if t != requester
                            and t not in current_nbs
                            and t not in self.isolated_nodes
                            and self.G.degree(t) > 0]

        if not possible_targets:
            return requester, None

        make_prompt = (
            "Make a new tie.\n"
            f"Tie-making ID list: {', '.join(possible_targets)}\n"
            "Please choose one player from tie-making list with whom to make a new tie.\n"
            "You will have the opportunity to connect with a new player even if you previously "
            "disconnected from that same player.\n"
            "If they agree, you will participate in future rounds with this player."
        )

        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[v]:
            msgs.append({"role": "user", "content": mem.get("phase", "")})
            msgs.append({"role": "user", "content": mem.get("decision", "")})
            msgs.append({"role": "assistant", "content": mem.get("decision_resp", "")})
            msgs.append({"role": "user", "content": mem.get("results", "")})
        msgs.append({"role": "user", "content": phase})
        msgs.append({"role": "user", "content": make_prompt})

        resp = self._call_llm(msgs).strip()
        chosen = None
        for line in resp.splitlines():
            for cand in possible_targets:
                if cand in line:
                    chosen = cand
                    break
            if chosen:
                break
        return requester, chosen

    def _process_single_accept(self, requester: str, target: str, system: str, phase: str):
        """Process acceptance decisions in parallel."""
        if target in self.isolated_nodes:
            return requester, target, False

        target_idx = self._id2idx[target]
        consent_prompt = (
            f"Participant {requester} requests to form a new tie with you. "
            f"Do you accept?"
            f"Please answer exactly: Accept or Reject."
        )

        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[target_idx]:
            msgs.append({"role": "user", "content": mem.get("phase", "")})
            msgs.append({"role": "user", "content": mem.get("decision", "")})
            msgs.append({"role": "assistant", "content": mem.get("decision_resp", "")})
            msgs.append({"role": "user", "content": mem.get("results", "")})
        msgs.append({"role": "user", "content": phase})
        msgs.append({"role": "user", "content": consent_prompt})

        resp = self._call_llm(msgs).strip()
        accepted = "accept" in resp.lower()
        return requester, target, accepted


    def _run_single_round(self, r: int):

        decision_network = self.G.copy()
        self.decision_networks.append(decision_network)
        self.edge_counts.append(decision_network.number_of_edges())
        safe_print(f"[Run {self.run_id}] Round {r + 1} start | Edges: {self.edge_counts[-1]}")

        system = self._system_prompt()
        phase = self._phase_prompt()


        decisions_all = {}
        memory_temp = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N) as executor:
            futures = {
                executor.submit(self._process_single_decision, v, aids, r, system, phase): aids
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
            self.decision_matrix[v][r] = decisions_all.get(aids, {})
            if aids in memory_temp:
                self.memory[v].append(memory_temp[aids])
            elif aids not in self.isolated_nodes:
                self.memory[v].append({"phase": phase, "decision": "", "decision_resp": "", "results": ""})


        for v, aids in enumerate(self.agent_ids):
            if aids in self.isolated_nodes or self.G.degree(aids) == 0:
                self.payoff_matrix[v, r] = 0
                continue

            neighbors = list(self.G.neighbors(aids))
            my_dec = decisions_all.get(aids, {})
            cost = sum(self.give_cost for d in my_dec.values() if d == "Give")

            rec = {}
            for nb in neighbors:
                rec[nb] = decisions_all.get(nb, {}).get(aids, "Keep")

            gain = sum(self.give_gain for d in rec.values() if d == "Give")
            payoff = gain - cost
            self.payoff_matrix[v, r] = payoff
            self.points[v] += payoff

            res_prompt = self._results_prompt(aids, my_dec, rec)
            if self.memory[v]:
                self.memory[v][-1]["results"] = res_prompt


        if (r + 1) % self.l_dyn == 0:
            self._rewiring_phase(decisions_all, system, phase)


        self._remove_isolated_nodes()


        self.networks_history.append(self.G.copy())
        self.edge_counts.append(self.G.number_of_edges())

    def _rewiring_phase(self, decisions_all: Dict, system: str, phase: str):

        to_cut = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N) as executor:
            futures = {
                executor.submit(self._process_single_cut, v, aids, decisions_all, system, phase): aids
                for v, aids in enumerate(self.agent_ids)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    res_aid, res_target = future.result()
                    if res_target:
                        to_cut.append((res_aid, res_target))
                except Exception:
                    pass


        cut_count = 0
        for agent, target in to_cut:
            if self.G.has_edge(agent, target):
                self.G.remove_edge(agent, target)
                cut_count += 1
        safe_print(f"[Run {self.run_id}] Cut {cut_count} edges")


        requesters = list(set(agent for agent, _ in to_cut))
        new_edge_requests = []

        if requesters:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(requesters)) as executor:
                futures = {
                    executor.submit(self._process_single_make, self._id2idx[req], req, system, phase): req
                    for req in requesters
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res_req, res_target = future.result()
                        if res_target:
                            new_edge_requests.append((res_req, res_target))
                    except Exception:
                        pass


        successful_edges = []
        if new_edge_requests:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(new_edge_requests)) as executor:
                futures = {
                    executor.submit(self._process_single_accept, req, tgt, system, phase): (req, tgt)
                    for req, tgt in new_edge_requests
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res_req, res_tgt, accepted = future.result()
                        if accepted:
                            successful_edges.append((res_req, res_tgt))
                    except Exception:
                        pass


        added_count = 0
        for u, v in successful_edges:
            if not self.G.has_edge(u, v) and u != v:
                self.G.add_edge(u, v)
                self.G[u][v]['dynamic'] = True
                added_count += 1
        safe_print(f"[Run {self.run_id}] Added {added_count} new edges")


        while self.G.number_of_edges() > self.edge_counts[-1]:
            if successful_edges:
                u, v = successful_edges.pop()
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)

    def _remove_isolated_nodes(self):
        new_isolated = {n for n in self.G.nodes() if self.G.degree(n) == 0}
        self.isolated_nodes.update(new_isolated)
        self.G.remove_nodes_from(new_isolated)

    def run_simulation(self):
        for r in range(self.L):
            self._run_single_round(r)
        return self._package_results()

    def _package_results(self):
        results_all = {
            "payoff_matrix": self.payoff_matrix,
            "decision_matrix": self.decision_matrix,
            "network_snapshots": self.networks_history,
            "final_points": self.points.copy(),
            "agent_ids": self.agent_ids,
            "give_ratio_per_round": [],
            "give_ratio_by_type_per_round": [],
            "edge_counts": self.edge_counts
        }


        for r in range(self.L):

            total_give = total_edges = 0
            for v in range(self.N):
                decisions = self.decision_matrix[v][r]
                total_edges += len(decisions)
                total_give += sum(1 for act in decisions.values() if act == "Give")
            ratio = total_give / total_edges if total_edges else 0
            results_all["give_ratio_per_round"].append((r + 1, ratio))


            static_give = static_total = 0
            dynamic_give = dynamic_total = 0
            G_snapshot = self.decision_networks[r]

            for v, aids in enumerate(self.agent_ids):
                if aids in self.isolated_nodes: continue
                for nb, action in self.decision_matrix[v][r].items():
                    if not G_snapshot.has_edge(aids, nb): continue

                    if G_snapshot[aids][nb].get('dynamic', False):
                        dynamic_total += 1
                        if action == "Give": dynamic_give += 1
                    else:
                        static_total += 1
                        if action == "Give": static_give += 1

            static_ratio = static_give / static_total if static_total else 0
            dynamic_ratio = dynamic_give / dynamic_total if dynamic_total else 0
            results_all["give_ratio_by_type_per_round"].append((r + 1, static_ratio, dynamic_ratio))

        return results_all


def run_single_experiment_set(i, alpha_list, N_num, L_num, p_num, memory_num, l_dyn_num, api_keys, model):
    """Run one complete replication across alpha values in a single thread."""
    safe_print(f"\n--- Starting Repeat {i}/10 ---")

    model_name_safe = model.replace("/", "_")
    dir_name = f"mixed_{model_name_safe}_repeat={i}"
    os.makedirs(dir_name, exist_ok=True)
    safe_print(f"[Repeat {i}] Results will be saved in: {dir_name}")

    for a in alpha_list:
        safe_print(f"[Repeat {i}] Running alpha = {a} ...")

        sim = MixedLLMCooperationSim(
            N=N_num, p=p_num, L=L_num, m=memory_num, l=l_dyn_num, alpha=a,
            models=model,
            give_cost=50,
            give_gain=100,
            api_interval=0.1,
            api_key=api_keys,
            initial_points=1000,
            run_id=i
        )
        results = sim.run_simulation()


        for rounds, g in enumerate(results["network_snapshots"]):
            gexf_filename = "mixed_net_r={}_N={}_L={}_p={}_m={}_l={}_a={}_mod={}.gexf".format(
                rounds, N_num, L_num, p_num, memory_num, l_dyn_num, a, model_name_safe
            )
            nx.write_gexf(g, os.path.join(dir_name, gexf_filename))

        csv_filename = 'mixed_pay_mat_N={}_L={}_p={}_m={}_l={}_a={}_mod={}.csv'.format(
            N_num, L_num, p_num, memory_num, l_dyn_num, a, model_name_safe
        )
        with open(os.path.join(dir_name, csv_filename), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Round'] + results['agent_ids']
            writer.writerow(header)
            for rounds in range(L_num):
                round_payoffs = results['payoff_matrix'][:, rounds]
                writer.writerow([rounds + 1] + list(round_payoffs))

        txt_filename = "mixed_dec_mat_N={}_L={}_p={}_m={}_l={}_a={}_mod={}.txt".format(
            N_num, L_num, p_num, memory_num, l_dyn_num, a, model_name_safe
        )
        with open(os.path.join(dir_name, txt_filename), "w") as f:
            for i_agent, aid in enumerate(results["agent_ids"]):
                f.write(f"Agent {aid}\n")
                for rounds in range(L_num):
                    f.write(f"  Round {rounds + 1}: {results['decision_matrix'][i_agent][rounds]}\n")


        log_msg = f"\n  --- Results for alpha = {a} (Repeat {i}) ---\n"
        log_msg += "Round | Give-ratio | Static Give | Dynamic Give\n"
        for i_round in range(L_num):
            round_num = i_round + 1
            total_rat = results["give_ratio_per_round"][i_round][1]
            static_rat = results["give_ratio_by_type_per_round"][i_round][1]
            dynamic_rat = results["give_ratio_by_type_per_round"][i_round][2]
            log_msg += f"{round_num:4}  | {total_rat:.3f}      | {static_rat:.3f}       | {dynamic_rat:.3f}\n"
        safe_print(log_msg)

    safe_print(f"--- Repeat {i} Complete ---")


if __name__ == "__main__":

    N_num = 20
    L_num = 15
    p_num = 0.3
    l_dyn_num = 3
    memory_num = 1
    alpha_num = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    api_keys = os.environ["LLM_API_KEY"]
    model = "gpt-5.2"

    print("Starting 10 parallel mixed experiments...")

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
