import numpy as np
import tqdm
from tqdm import tqdm
import time

neutron_energies = np.linspace(0.01, 5.5, 66)
iterations = 10

for energy in tqdm(neutron_energies):
    time.sleep(0.1)
    print("Testing" + str(energy))
    for iteration in tqdm(range(iterations)):
        time.sleep(0.2)
        print("Waiting")
