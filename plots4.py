import re
import statistics
import matplotlib.pyplot as plt
import numpy as np

files = [("output_time_1.out", 1), ("output_time_2.out", 2),  ("output_time_4.out", 4),  ("output_time_10.out", 10)]

def get_metrics(filename):
  with open(filename, "r") as f:
    content = f.read()
    lines = content.split("\n")
    processes_vs_times_map = {}
    pattern = "TIME: (\d+) ACC: (\d+\.\d+)"
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
  
  
  return indices, S

for file, N in files:
  indices, S = get_metrics(file)
  plt.scatter(indices, S.values(), label = f"P={N}" )
plt.title("Wykres zależności skuteczności ACC(t) \n od czasu, dla różnej liczby porcesów P")
plt.xlabel("Czas od początku uczenia [s]")
plt.ylabel("Skuteczność klasyfikacji na zbiorze walidacyjnym ACC(t)")
plt.legend()
plt.show()
