
"""
Static‐network LLM‐driven simulation
"""

import os
import csv
import time
import string
import numpy as np
import networkx as nx
from openai import OpenAI
from collections import deque
from typing import List, Dict, Tuple
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


class StaticLLMCooperationSim:
    def __init__(
            self,
            N: int,
            p: float,
            L: int,
            m: int,
            api_key: str = "",
            base_url: str = "https://www.dmxapi.cn/v1",
            models: str = "gemini-3-flash-preview",
            initial_points: int = 1000,
            give_cost: int = 50,
            give_gain: int = 100,
            api_interval: float = 1.0,
            run_id: int = 0
    ):

        self.N, self.p, self.L, self.m = N, p, L, m
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


        isolated_nodes = [n for n in self.G.nodes() if self.G.degree(n) == 0]
        self.G.remove_nodes_from(isolated_nodes)


        self.agent_ids = list(self.G.nodes())
        self.N = len(self.agent_ids)


        self.points = np.full(self.N, initial_points, dtype=int)
        self.payoff_matrix = np.zeros((self.N, L), dtype=int)

        self.decision_matrix = [[{} for _ in range(L)] for _ in range(self.N)]
        self.memory = [deque(maxlen=m if m != -1 else None) for _ in range(self.N)]


        self._id2idx = {aids: v for v, aids in enumerate(self.agent_ids)}


    def _system_prompt(self) -> str:
        """Generate the system prompt."""
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
            f"If the other chooses {{Give}}, the other pays {self.give_cost} points for you to gain {self.give_gain} points. "
            f"If the other chooses {{Keep}}, you don't gain any points.\n"
            "Once everyone makes their decisions, the results are displayed. "
            "You will be shown the choices of each other you are connected to, and how many points in total you "
            "gained or lost.\n"
            "Remember, you will earn the number of points in one randomly selected phase of the study."
        )

    def _phase_prompt(self) -> str:
        """Generate the stage prompt."""
        return (
            "In this phase, in all rounds, after you learn the results, you will continue interacting with the same "
            "others you were connected to in the previous round."
            "This indicates that you will be connected to the other throughout the remainder of this phase.\n"
            "Be sure you have carefully read the instructions."
        )

    def _decision_prompt_base(self, agent_id: str, current_points: int, round_num: int) -> str:
        """Generate the decision prompt."""
        neighbors = list(self.G.neighbors(agent_id))
        neighbor_str = ", ".join(neighbors) if neighbors else "None"


        first_round_prompt = ""
        if round_num == 0:
            first_round_prompt = (
                "\nThis is the first round. To ensure diverse initial strategies. "
                "Please mix Give and Keep choices among your neighbors. "
                "Randomly select some neighbors to Give and others to Keep."
            )

        return (
            f"Your ID: {agent_id}, Your point: {current_points}\n"
            f"Neighbor ID list: {neighbor_str}\n"
            "Please choose to {Give} or {Keep}:\n"
            f"\tIf you choose {{Give}} you will pay {self.give_cost} points to give your neighbor {self.give_gain} points\n"
            f"\tIf you choose {{Keep}} you will pay {{0}} points to give your neighbor {{0}} points\n"
            "Do you want to choose {Give} or {Keep} with this neighbor?\n\n"
            "**Response Format**\n"
            "```plaintext\n"
            "For neighbor {ID}, I choose {ACTION}\n"
            "```"
            f"{first_round_prompt}"
        )

    def _results_prompt(self, agent_id: str, decisions: Dict[str, str], received: Dict[str, str]) -> str:
        """Generate the outcome prompt."""
        give_dec = [n for n, d in decisions.items() if d == "Give"]
        keep_dec = [n for n, d in decisions.items() if d == "Keep"]
        give_rec = [n for n, d in received.items() if d == "Give"]
        keep_rec = [n for n, d in received.items() if d == "Keep"]

        give_point_neb = len(give_rec) * self.give_gain if give_rec else 0

        return (
            "Results:\n"
            f" You chose {{Give}} with {len(give_dec)} player(s)—{','.join(give_dec)}—and paid "
            f"a total of {self.give_cost * len(give_dec)} points to give {self.give_gain} points to each of them.\n"
            f" You chose {{Keep}} with {len(keep_dec)} player(s)—{','.join(keep_dec)}—and paid "
            f"{{0}} points to give {{0}} points to each of them.\n"
            f" {len(give_rec)} player(s) you are connected with chose {{Give}}—{','.join(give_rec)}—and gave you a "
            f"total of {give_point_neb} points.\n"
            f"{len(keep_rec)} player(s) you are connected with chose {{Keep}}—{','.join(keep_rec)}—and gave you {{0}} "
            f"points."
        )


    def _parse_decisions(self, text: str, neighbors: List[str]) -> Dict[str, str]:
        """
        Parse the LLM's decision response.
        """
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
        """
        Call the LLM API.
        """
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

    def _process_single_agent(self, v: int, aids: str, r: int, system_prompt: str, phase_prompt: str) -> Tuple[
        str, Dict[str, str], Dict]:
        """
        Process one agent decision for parallel execution.
        """

        msgs = [{"role": "system", "content": system_prompt}]
        for mem in self.memory[v]:
            msgs.append({"role": "user", "content": mem["phase"]})
            msgs.append({"role": "user", "content": mem["decision"]})
            msgs.append({"role": "assistant", "content": mem["decision_resp"]})
            msgs.append({"role": "user", "content": mem["results"]})


        decision_prompt = self._decision_prompt_base(aids, self.points[v], r)
        msgs.append({"role": "user", "content": phase_prompt})
        msgs.append({"role": "user", "content": decision_prompt})


        resp = self._call_llm(msgs)
        neighbors = list(self.G.neighbors(aids))
        decision_result = self._parse_decisions(resp, neighbors)


        memory_entry = {
            "phase": phase_prompt,
            "decision": decision_prompt,
            "decision_resp": "\n".join([f"For neighbor {n}, I choose {d}"
                                        for n, d in decision_result.items()]),
            "results": "",
        }

        return aids, decision_result, memory_entry


    def _run_single_round(self, r: int):
        """
        Run one simulation round with parallel agent decisions.
        """
        system = self._system_prompt()
        phase = self._phase_prompt()


        decisions_all: Dict[str, Dict[str, str]] = {}
        memory_temp = {}


        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N) as executor:
            future_to_agent = {
                executor.submit(self._process_single_agent, v, aids, r, system, phase): (v, aids)
                for v, aids in enumerate(self.agent_ids)
            }

            for future in concurrent.futures.as_completed(future_to_agent):
                v, aids = future_to_agent[future]
                try:
                    res_aids, decision_res, mem_entry = future.result()
                    decisions_all[res_aids] = decision_res
                    memory_temp[res_aids] = mem_entry
                    self.decision_matrix[v][r] = decision_res.copy()
                except Exception as exc:
                    safe_print(f"[Run {self.run_id}] Agent {aids} generated an exception: {exc}")

                    neighbors = list(self.G.neighbors(aids))
                    fallback_dec = {n: "Keep" for n in neighbors}
                    decisions_all[aids] = fallback_dec
                    self.decision_matrix[v][r] = fallback_dec

                    memory_temp[aids] = {
                        "phase": phase,
                        "decision": self._decision_prompt_base(aids, self.points[v], r),
                        "decision_resp": "Error",
                        "results": ""
                    }


        for v, aids in enumerate(self.agent_ids):
            neighbors = list(self.G.neighbors(aids))
            my_dec = decisions_all.get(aids, {n: "Keep" for n in neighbors})


            cost = sum(self.give_cost for d in my_dec.values() if d == "Give")


            rec = {}
            for nb in neighbors:

                neighbor_decisions = decisions_all.get(nb, {})
                rec[nb] = neighbor_decisions.get(aids, "Keep")

            gain = sum(self.give_gain for d in rec.values() if d == "Give")


            payoff = gain - cost
            self.payoff_matrix[v, r] = payoff
            self.points[v] += payoff

            res_prompt = self._results_prompt(aids, my_dec, rec)

            if aids in memory_temp:
                memory_temp[aids]["results"] = res_prompt
                self.memory[v].append(memory_temp[aids])


    def run_simulation(self):
        """
        Run the complete simulation.
        """
        for r in range(self.L):
            safe_print(f"[Run {self.run_id}] --- Round {r + 1}/{self.L} ---")
            self._run_single_round(r)
        return self._package_results()

    def _package_results(self):
        """
        Package the simulation results.
        """

        give_ratio = np.zeros((self.L, 2))
        for r in range(self.L):
            give_cnt = total_cnt = 0
            for v, aids in enumerate(self.agent_ids):
                for nb, act in self.decision_matrix[v][r].items():
                    total_cnt += 1
                    if act == "Give":
                        give_cnt += 1
            give_ratio[r, 0] = r + 1
            give_ratio[r, 1] = give_cnt / total_cnt if total_cnt else 0.0

        return {
            "payoff_matrix": self.payoff_matrix,
            "decision_matrix": self.decision_matrix,
            "network": self.G.copy(),
            "give_ratio_per_round": give_ratio,
            "final_points": self.points.copy(),
            "agent_ids": self.agent_ids,
        }


def run_single_experiment(i, N_num, L_num, p_num, memory_num, api_keys, model):
    """
    Run one complete experiment for parallel execution.
    """
    safe_print(f"\n--- Starting Run {i}/10 ---")

    model_name_safe = model.replace("/", "_")
    dir_name = f"static_{model_name_safe}_repeat={i}"
    os.makedirs(dir_name, exist_ok=True)
    safe_print(f"[Run {i}] Results will be saved in: {dir_name}")

    sim = StaticLLMCooperationSim(
        N=N_num, p=p_num, L=L_num, m=memory_num,
        models=model,
        give_cost=50,
        give_gain=100,
        api_interval=0.1,
        api_key=api_keys,
        initial_points=1000,
        run_id=i
    )
    results = sim.run_simulation()

    gexf_filename = "static_net_N={}_L={}_p={}_m={}_mod={}.gexf".format(
        N_num, L_num, p_num, memory_num, model_name_safe
    )
    csv_filename = 'static_pay_mat_N={}_L={}_p={}_m={}_mod={}.csv'.format(
        N_num, L_num, p_num, memory_num, model_name_safe
    )
    txt_filename = "static_dec_mat_N={}_L={}_p={}_m={}_mod={}.txt".format(
        N_num, L_num, p_num, memory_num, model_name_safe
    )

    gexf_path = os.path.join(dir_name, gexf_filename)
    csv_path = os.path.join(dir_name, csv_filename)
    txt_path = os.path.join(dir_name, txt_filename)


    nx.write_gexf(results["network"], gexf_path)


    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Round'] + results['agent_ids']
        writer.writerow(header)
        for r in range(L_num):
            round_payoffs = results['payoff_matrix'][:, r]
            writer.writerow([r + 1] + list(round_payoffs))


    with open(txt_path, "w") as f:
        for i_agent, aid in enumerate(results["agent_ids"]):
            f.write(f"Agent {aid}\n")
            for rounds in range(L_num):
                f.write(f"  Round {rounds + 1}: {results['decision_matrix'][i_agent][rounds]}\n")


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
    memory_num = 1
    api_keys = os.environ["LLM_API_KEY"]
    model = "gemini-3-flash-preview"

    print("Starting 10 parallel experiments...")


    max_parallel_experiments = 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_experiments) as executor:
        futures = []
        for i in range(1, 11):
            futures.append(
                executor.submit(run_single_experiment, i, N_num, L_num, p_num, memory_num, api_keys, model)
            )


        concurrent.futures.wait(futures)

    print("\n--- All 10 runs complete. ---")
