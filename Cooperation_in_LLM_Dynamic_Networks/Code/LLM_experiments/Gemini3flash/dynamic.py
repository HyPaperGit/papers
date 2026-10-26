
"""
Dynamic‐network LLM‐driven simulation
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


class DynamicLLMCooperationSim:
    def __init__(
            self,
            N: int,
            p: float,
            L: int,
            m: int,
            l: int = 3,
            api_key: str = "",
            base_url: str = "https://www.dmxapi.cn/v1",
            models: str = "gemini-3-flash-preview",
            initial_points: int = 1000,
            give_cost: int = 50,
            give_gain: int = 100,
            api_interval: float = 1.0,
            run_id: int = 0
    ):
        self.N, self.p, self.L, self.m, self.l_dyn = N, p, L, m, l
        self.model = models
        self.api_key = api_key
        self.base_url = base_url
        self.give_cost = give_cost
        self.give_gain = give_gain
        self.api_interval = api_interval
        self.initial_points = initial_points
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.run_id = run_id


        self.agent_ids = generate_agent_ids(N)

        max_edges = N * (N - 1) // 2
        M = round(p * max_edges)
        self.G = nx.gnm_random_graph(N, M)
        self.G = nx.relabel_nodes(self.G, dict(zip(range(N), self.agent_ids)))
        self.networks_history: List[nx.Graph] = [self.G.copy()]


        self.memory = [deque(maxlen=m) for _ in range(N)]
        self.points = np.full(N, initial_points, dtype=int)
        self.payoff_matrix = np.zeros((N, L), dtype=int)

        self.decision_matrix = [[{} for _ in range(L)] for _ in range(N)]


        self._id2idx = {aids: v for v, aids in enumerate(self.agent_ids)}


    def _system_prompt(self) -> str:
        return (
            "Welcome in today's study, you will be making decisions with other participants.\n"
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
            f"If the other chooses Keep, you don't gain any points.\n"
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
            "You can stop interacting with any one of the others you are connected to. "
            "You will see an ID list of participants allowed to disconnect. "
            "This indicates that, in some rounds, you will be able to disconnected from the other.\n"
            "While you are deciding whether and with whom you wish to stop interacting, "
            "the others will be deciding whether they wish to continue interacting with their ties, including you.\n"
            "In those rounds that you are able to, if you choose to stop interacting with one of the others, "
            "you will choose which one new other you want to connect with.\n"
            "Specifically, you will see a list containing the Participant IDs of all of the new others with whom you "
            "could potentially connect. From the list, you can select one other to connect to.\n"
            "You will have the opportunity to connect with a new player even if you previously "
            "disconnected from that same player.\n"
            "The other must confirm this tie in order to finalize it. Thus, the other will be asked if he or she would "
            "like to connect with you. If they do, you will successfully be connected with the new other.\n"
            "Similarly, you will be given an opportunity to confirm any ties others want to form with you.\n"
            "Be sure you have carefully read the instructions."
        )

    def _first_round_instruction_prompt(self) -> str:
        return (
            "\nThis is the first round. To ensure diverse initial strategies. "
            "Please mix Give and Keep choices among your neighbors. "
            "Randomly select some neighbors to Give and others to Keep."
        )

    def _decision_prompt_base(self, agent_id: str, current_points: int, is_first_round: bool = False) -> str:
        neighbors = list(self.G.neighbors(agent_id))
        neighbor_str = ", ".join(neighbors) if neighbors else "None"
        first_round_instruction = self._first_round_instruction_prompt() if is_first_round else ""
        return (
            f"Your ID: {agent_id}, Your point: {current_points}\n"
            f"Neighbor ID list: {neighbor_str}\n"
            f"{first_round_instruction}\n"
            "Please choose to Give or Keep:\n"
            f"\tIf you choose Give you will pay {self.give_cost} points to give your neighbor {self.give_gain} points\n"
            f"\tIf you choose Keep you will pay 0 points to give your neighbor 0 points\n"
            "Do you want to choose Give or Keep with this neighbor?\n\n"
            "**Response Format**\n"
            "```plaintext\n"
            "For neighbor {ID}, I choose {ACTION}\n"
            "```"
        )

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

                time.sleep(2 ** attempt)
        return ""


    def _process_single_decision(self, v: int, aids: str, rounds: int, system: str, phase: str):
        """Process decisions in parallel."""
        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[v]:
            msgs.append({"role": "user", "content": mem["phase"]})
            msgs.append({"role": "user", "content": mem["decision"]})
            msgs.append({"role": "assistant", "content": mem["decision_resp"]})
            msgs.append({"role": "user", "content": mem["results"]})

        decision_prompt = self._decision_prompt_base(aids, self.points[v], is_first_round=(rounds == 0))
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
        """Process tie-removal intentions in parallel."""
        neighbors = list(self.G.neighbors(aids))
        if not neighbors:
            return aids, None

        give_by_nbs = [n for n in neighbors if decisions_all.get(n, {}).get(aids, "Keep") == "Give"]
        keep_by_nbs = [n for n in neighbors if decisions_all.get(n, {}).get(aids, "Keep") == "Keep"]
        cut_list_str = ", ".join(neighbors)

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
            if "no player" in line.lower() or "none" in line.lower():
                chosen = None
                break
            for nb in neighbors:
                if nb in line:
                    chosen = nb
                    break
        return aids, chosen

    def _process_single_make(self, v: int, req: str, decisions_all: Dict, system: str, phase: str):
        """Process partner-selection intentions in parallel for agents that removed ties."""
        current_nbs = set(self.G.neighbors(req))
        possible = [t for t in self.agent_ids
                    if t != req and t not in current_nbs and self.G.degree(t) > 0]

        if not possible:
            return req, None

        poss_with_action = []
        for cand in possible:
            act = decisions_all.get(cand, {}).get(req, "Keep")
            poss_with_action.append(f"{cand}({act})")
        poss_str = ", ".join(poss_with_action)

        make_prompt = (
            "Make a new tie.\n"
            f"Tie-making ID list: {poss_str}\n"
            "Please choose one player from tie-making list with whom to make a new tie.\n"
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
        for cand in possible:
            if cand in resp:
                chosen = cand
                break
        return req, chosen

    def _process_single_accept(self, requester: str, requested: str, system: str, phase: str):
        """Process acceptance and rejection decisions in parallel."""
        target_idx = self._id2idx[requested]
        consent_prompt = (
            f"Participant {requester} requests to form a new tie "
            f"with you. Do you accept?"
            f"Please answer exactly: Accept or Reject."
        )

        msgs = [{"role": "system", "content": system}]
        for mem in self.memory[target_idx]:
            msgs.append({"role": "user", "content": mem["phase"]})
            msgs.append({"role": "user", "content": mem["decision"]})
            msgs.append({"role": "assistant", "content": mem["decision_resp"]})
            msgs.append({"role": "user", "content": mem["results"]})

        msgs.append({"role": "user", "content": phase})
        msgs.append({"role": "user", "content": consent_prompt})

        resp = self._call_llm(msgs).strip()
        return requester, requested, "accept" in resp.lower()


    def _run_single_round(self, rounds: int):
        system = self._system_prompt()
        phase = self._phase_prompt()


        decisions_all: Dict[str, Dict[str, str]] = {}
        memory_temp = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N) as executor:
            futures = {
                executor.submit(self._process_single_decision, v, aids, rounds, system, phase): (v, aids)
                for v, aids in enumerate(self.agent_ids)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    res_aids, res_dec, res_mem = future.result()
                    decisions_all[res_aids] = res_dec
                    memory_temp[res_aids] = res_mem
                except Exception as e:
                    safe_print(f"[Run {self.run_id}] Decision Error: {e}")


        for v, aids in enumerate(self.agent_ids):

            if aids in decisions_all:
                self.decision_matrix[v][rounds] = decisions_all[aids]


            if aids in memory_temp:
                self.memory[v].append(memory_temp[aids])
            else:

                self.memory[v].append({"phase": phase, "decision": "", "decision_resp": "Error", "results": ""})


        for v, aids in enumerate(self.agent_ids):
            neighbors = list(self.G.neighbors(aids))
            my_dec = decisions_all.get(aids, {n: "Keep" for n in neighbors})
            cost = sum(self.give_cost for d in my_dec.values() if d == "Give")

            rec = {}
            for nb in neighbors:
                rec[nb] = decisions_all.get(nb, {}).get(aids, "Keep")

            gain = sum(self.give_gain for d in rec.values() if d == "Give")
            payoff = gain - cost
            self.payoff_matrix[v, rounds] = payoff
            self.points[v] += payoff

            res_prompt = self._results_prompt(aids, my_dec, rec)
            self.memory[v][-1]["results"] = res_prompt


        if (rounds + 1) % self.l_dyn == 0:
            self._rewiring_phase(decisions_all, system, phase)


        self.networks_history.append(self.G.copy())


    def _rewiring_phase(self, decisions_all: Dict[str, Dict[str, str]], system: str, phase: str):

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
                except Exception as e:
                    pass


        edges_to_remove = []
        for agent, neighbor in to_cut.items():
            if self.G.has_edge(agent, neighbor):
                edges_to_remove.append((agent, neighbor))
        self.G.remove_edges_from(edges_to_remove)


        requesters = list(to_cut.keys())
        new_edge_requests: Dict[str, str] = {}

        if requesters:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(requesters)) as executor:
                futures = {
                    executor.submit(self._process_single_make, self._id2idx[req], req, decisions_all, system,
                                    phase): req
                    for req in requesters
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res_req, res_target = future.result()
                        if res_target:
                            new_edge_requests[res_req] = res_target
                    except Exception:
                        pass


        edges_to_add = []
        processed_pairs = set()


        accept_tasks = []
        for requester, requested in new_edge_requests.items():
            if (requester, requested) in processed_pairs or (requested, requester) in processed_pairs:
                continue
            processed_pairs.add((requester, requested))
            accept_tasks.append((requester, requested))

        if accept_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(accept_tasks)) as executor:
                futures = {
                    executor.submit(self._process_single_accept, req, tgt, system, phase): (req, tgt)
                    for req, tgt in accept_tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        r_req, r_tgt, accepted = future.result()
                        if accepted:
                            if not self.G.has_edge(r_req, r_tgt):
                                edges_to_add.append((r_req, r_tgt))
                    except Exception:
                        pass


        self.G.add_edges_from(edges_to_add)


    def run_simulation(self):
        prev_edges = self.G.number_of_edges()
        for rounds in range(self.L):
            safe_print(f"[Run {self.run_id}] --- Round {rounds + 1}/{self.L}  (edges: {self.G.number_of_edges()}) ---")
            self._run_single_round(rounds)


            current_edges = self.G.number_of_edges()


            prev_edges = current_edges

        return self._package_results()

    def _package_results(self):
        give_ratio = np.zeros((self.L, 2))
        for rounds in range(self.L):
            give_cnt = total_cnt = 0
            for v in range(self.N):
                dec_dict = self.decision_matrix[v][rounds]
                for act in dec_dict.values():
                    total_cnt += 1
                    if act == "Give":
                        give_cnt += 1

            give_ratio[rounds, 0] = rounds + 1
            give_ratio[rounds, 1] = give_cnt / total_cnt if total_cnt else 0.0

        return {
            "payoff_matrix": self.payoff_matrix,
            "decision_matrix": self.decision_matrix,
            "network_snapshots": self.networks_history,
            "give_ratio_per_round": give_ratio,
            "final_points": self.points.copy(),
            "agent_ids": self.agent_ids,
        }


def run_single_experiment(i, N_num, L_num, p_num, memory_num, l_dyn_num, api_keys, model):
    safe_print(f"\n--- Starting Run {i}/10 ---")

    model_name_safe = model.replace("/", "_")
    dir_name = f"dynamic_{model_name_safe}_repeat={i}"
    os.makedirs(dir_name, exist_ok=True)
    safe_print(f"[Run {i}] Results will be saved in: {dir_name}")

    sim = DynamicLLMCooperationSim(
        N=N_num, p=p_num, L=L_num, m=memory_num, l=l_dyn_num,
        models=model,
        give_cost=50,
        give_gain=100,
        api_interval=0.1,
        api_key=api_keys,
        initial_points=1000,
        run_id=i
    )

    results = sim.run_simulation()


    for r, g in enumerate(results["network_snapshots"]):
        gexf_filename = "dyn_net_r={}_N={}_L={}_p={}_m={}_l={}_mod={}.gexf".format(
            r, N_num, L_num, p_num, memory_num, l_dyn_num, model_name_safe
        )
        gexf_path = os.path.join(dir_name, gexf_filename)
        nx.write_gexf(g, gexf_path)


    csv_filename = 'dyn_pay_mat_N={}_L={}_p={}_m={}_l={}_mod={}.csv'.format(
        N_num, L_num, p_num, memory_num, l_dyn_num, model_name_safe
    )
    csv_path = os.path.join(dir_name, csv_filename)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Round'] + results['agent_ids']
        writer.writerow(header)
        for r in range(L_num):
            round_payoffs = results['payoff_matrix'][:, r]
            writer.writerow([r + 1] + list(round_payoffs))


    txt_filename = "dyn_dec_mat_N={}_L={}_p={}_m={}_l={}_mod={}.txt".format(
        N_num, L_num, p_num, memory_num, l_dyn_num, model_name_safe
    )
    txt_path = os.path.join(dir_name, txt_filename)
    with open(txt_path, "w") as f:
        for i_agent, aid in enumerate(results["agent_ids"]):
            f.write(f"Agent {aid}\n")
            for r in range(L_num):
                f.write(f"  Round {r + 1}: {results['decision_matrix'][i_agent][r]}\n")


    safe_print(f"\n--- Results for Run {i}/10 ---")
    log_str = "Round | Give-ratio (edges)\n"
    for row in results["give_ratio_per_round"]:
        log_str += f"{int(row[0]):4}  | {row[1]:.3f}\n"
    safe_print(log_str)
    safe_print(f"--- Run {i} Complete ---")


if __name__ == "__main__":

    N_num = 20
    L_num = 15
    p_num = 0.3
    l_dyn_num = 3
    memory_num = 1
    api_keys = os.environ["LLM_API_KEY"]
    model = "gemini-3-flash-preview"

    print("Starting 10 parallel dynamic experiments...")

    max_parallel_experiments = 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_experiments) as executor:
        futures = []
        for i in range(1, 11):
            futures.append(
                executor.submit(run_single_experiment, i, N_num, L_num, p_num, memory_num, l_dyn_num, api_keys, model)
            )

        concurrent.futures.wait(futures)

    print("\n--- All 10 runs complete. ---")
