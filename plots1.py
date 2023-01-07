import re
import statistics
import matplotlib.pyplot as plt
import numpy as np

files = [("output1.out", 1), ("output3.out", 3)]

def get_metrics(filename):
  with open(filename, "r") as f:
    content = f.read()
    lines = content.split("\n")
    processes_vs_times_map = {}
    pattern = "PROCESSES: (\d+) TIME: (\d+\.\d+)"
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

  S = {p: T[1]/T[p] for p in T.keys()}
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
  plt.plot(indices, S.values(), label = f"N={N}" )
  plt.fill_between(indices, lower_bound, upper_bound, alpha=0.3)
plt.title("Wykres zależności przyśpieszenia S(p) \n od liczby procesów p, dla ustalonej liczby epok N")
plt.xlabel("Liczba procesów p")
plt.ylabel("Przyśpieszenie S(p)")
plt.legend()
plt.show()

for file, N in files:
  indices, S, S_std, E, E_std, f, f_std = get_metrics(file)
  lower_bound = [x - y for x, y in zip(E.values(), E_std.values())]
  upper_bound = [x + y for x, y in zip(E.values(), E_std.values())]
  plt.plot(indices, E.values(), label = f"N={N}" )
  plt.fill_between(indices, lower_bound, upper_bound, alpha=0.3)
plt.title("Wykres zależności efektywności E(p) \n od liczby procesów p, dla ustalonej liczby epok N")
plt.xlabel("Liczba procesów p")
plt.ylabel("Efektywność E(p)")
plt.legend()
plt.show()
