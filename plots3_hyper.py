import re
import statistics
import matplotlib.pyplot as plt
import numpy as np

files = [("output_lr_00025.out", "0.00025(!)"), ("output_lr_00434.out", "0.00434"), ("output_lr_00049.out", "0.00049"), ("output_lr_default.out", "0.01 (def)")]

def get_metrics(filename):
  with open(filename, "r") as f:
    content = f.read()
    lines = content.split("\n")
    processes_vs_times_map = {}
    pattern = "EPOCH: (\d+) ACC: (\d+\.\d+) LOSS: (\d+\.\d+) TIME: (\d+)"
    for line in lines:
      match = re.match(pattern, line)
      number_of_processes = int(match.group(1))
      time = float(match.group(3))
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
  plt.plot(indices, S.values(), label = f"lr={N}" )
plt.title("Wykres zależności funkcji kosztu LOSS(e) \n od numeru epoki treningowej e, dla różnych parametrów lr")
plt.xlabel("Numer epoki treningowej e")
plt.ylabel("Wartość funkcji kosztu LOSS(e)")
plt.legend()
plt.show()
