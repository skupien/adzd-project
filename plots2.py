import re
import statistics
import matplotlib.pyplot as plt
import numpy as np

files = [("output_acc_1.out", 1), ("output_acc_5.out", 5), ("output_acc_10.out", 10)]

def get_metrics(filename):
  with open(filename, "r") as f:
    content = f.read()
    lines = content.split("\n")
    processes_vs_times_map = {}
    pattern = "EPOCH: (\d+) ACC: (\d+\.\d+)"
    for line in lines:
      match = re.match(pattern, line)
      number_of_processes = int(match.group(1))
      time = float(match.group(2))
      if number_of_processes not in processes_vs_times_map:
        processes_vs_times_map[number_of_processes] = []
      processes_vs_times_map[number_of_processes].append(time)
  
  for p in processes_vs_times_map.keys():
    processes_vs_times_map[p] = np.array(processes_vs_times_map[p])
  indices = list(processes_vs_times_map.keys())
  T = processes_vs_times_map

  S = {p: T[p] for p in T.keys()}
  S_avg = {p: np.average(S[p]) for p in T.keys()}
  S_std = {p: statistics.stdev(S[p]) for p in T.keys()}
  
  E = {p: S[p]/p for p in T.keys()}
  E_avg = {p: np.average(E[p]) for p in T.keys()}
  E_std = {p: statistics.stdev(E[p]) for p in T.keys()}
  
  f = {p: (1/S[p]-1/p)/(1-1/p) for p in T.keys() if p != 1}
  f_avg = {p: np.average(f[p]) for p in T.keys() if p != 1}
  f_std = {p: statistics.stdev(f[p]) for p in T.keys() if p != 1}
  
  return indices, S_avg, S_std, E_avg, E_std, f_avg, f_std

for file, N in files:
  indices, S, S_std, E, E_std, f, f_std = get_metrics(file)
  lower_bound = [x - y for x, y in zip(S.values(), S_std.values())]
  upper_bound = [x + y for x, y in zip(S.values(), S_std.values())]
  plt.plot(indices, S.values(), label = f"P={N}" )
  plt.fill_between(indices, lower_bound, upper_bound, alpha=0.15)
plt.title("Wykres zależności ACC(e) \n od numeru epoki treningowej e, dla ustalonej liczby procesów P")
plt.xlabel("Numer epoki treningowej e")
plt.ylabel("Skuteczność klasyfikacji na zbiorze walidacyjnym ACC(e)")
plt.legend()
plt.show()
