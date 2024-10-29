import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
import time
import bisect
from IPython.display import display, clear_output


def scheduler(T, rate_list, resolution):
    upper_bound_rate = np.max(rate_list)
    cumu_time = 0
    happen_time_step = [] # each element means at what time does an event happens
    while cumu_time < T:
        wait = np.random.exponential(scale=1/upper_bound_rate)
        cumu_time += wait
        happen_time_step.append(cumu_time)
    happen_time_step = np.array(happen_time_step)
    accept = np.zeros_like(happen_time_step)
    for i in range(happen_time_step.shape[0]):
        t = happen_time_step[i]
        step = int(t / resolution)
        if step >= len(rate_list):
            step = len(rate_list) - 1
        rate = rate_list[step]
        if np.random.uniform(0, 1) < rate / upper_bound_rate:
            accept[i] = 1
    return happen_time_step[accept==1]


def plotcell(states, center):
    # figure initialization
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.plot(0, 0, 0, 'ro')
    x_data = np.zeros(0)
    y_data = np.zeros(0)
    z_data = np.zeros(0)
    sc = ax.scatter(x_data, y_data, z_data)
    # plot states at all the time steps
    for t in range(len(states)):
        molecules = states[t]
        num_molecules = len(molecules)
        x_data = np.zeros(num_molecules)
        y_data = np.zeros(num_molecules)
        z_data = np.zeros(num_molecules)
        for i in range(num_molecules):
            x_data[i], y_data[i], z_data[i], _ = molecules[i]
        # dynamically plot the diffusion process
        sc._offsets3d = (x_data, y_data, z_data)  # Update scatter plot data
        ax.set_title(f'Iteration {t+1}')
        display(fig)
        time.sleep(0.2)
        clear_output(wait=True)


def build_data(states, radius, num_dists):
    num_steps = len(states)
    data = np.zeros((num_steps, num_dists))
    stride = radius / data.shape[1]
    for i in range(num_steps):
        molecules = states[i]
        for m in range(len(molecules)):
            x, y, z, dist = molecules[m]
            j = int(dist / stride)
            data[i][j] += 1
    return data


class Molecule:
    def __init__(self, id, velocity, sigma, t):
        self.id = id
        self.born_time = t
        self.distance = 0
        self.velocity = velocity
        self.sigma = sigma
        self.phi = np.random.uniform(0, math.pi * 2, size=1)  # z-axis angle
        self.theta = np.random.uniform(0, math.pi * 2, size=1)# xy-axis angle


class Cell:
    def __init__(self, center, radius, resolution, velo, brownian):
        self.center = center
        self.radius = radius
        self.molecules = []
        self.maxid = 0
        self.resolution = resolution
        self.degrade_rate = 0.5
        # self.velocity = np.random.uniform(0.001, 0.02)
        # self.sigma = np.random.uniform(0, 0.01)
        self.velocity = velo
        self.sigma = brownian
        self.trajectory = []

    def transcription(self, t):
        molecule = Molecule(self.maxid+1, self.velocity, self.sigma, t)
        self.molecules.append(molecule)
        self.maxid += 1

    def diffuse(self, t, schedule):
        # Find the sublist of shcedule with values between (t-resolution) ~ t
        left_index = bisect.bisect_left(schedule, t-self.resolution)
        right_index = bisect.bisect_left(schedule, t)
        generated = schedule[left_index:right_index]
        for t in generated:
            self.transcription(t)
        # Diffusion
        for molecule in self.molecules:
            location = molecule.velocity * (t - molecule.born_time)
            molecule.distance = np.random.normal(location, molecule.sigma)
            molecule.phi = np.random.normal(molecule.phi, 0.01)
            molecule.theta = np.random.normal(molecule.theta, 0.01)

    def degrade(self):
        degrade_list = []
        for molecule in self.molecules:
            if molecule.distance > 0.9 * self.radius:
                hit = np.random.binomial(1, self.degrade_rate)
                if hit == 1:
                    degrade_list.append(molecule.id)
        self.molecules = [molecule for molecule in self.molecules if molecule.id not in degrade_list]

    def dynamic(self, T, schedule):
        # run the diffusion process and save points at each time step
        x_data = np.zeros(0)
        y_data = np.zeros(0)
        z_data = np.zeros(0)
        time_steps = np.arange(0, T, self.resolution)
        # self.trajectory.append(self.molecules)
        for t in range(time_steps.shape[0]):
            self.diffuse(time_steps[t], schedule)
            self.degrade()
            # self.trajectory.append(self.molecules) # record the current state
            num_molecules = len(self.molecules)
            x_data = np.zeros(num_molecules)
            y_data = np.zeros(num_molecules)
            z_data = np.zeros(num_molecules)
            state = []
            for i in range(num_molecules):
                dist = self.molecules[i].distance
                phi = self.molecules[i].phi
                theta = self.molecules[i].theta
                x = dist * math.sin(phi) * math.cos(theta) + self.center[0]
                y = dist * math.sin(phi) * math.sin(theta) + self.center[1]
                z = dist * math.cos(phi) + self.center[2]
                x_data[i] = x
                y_data[i] = y
                z_data[i] = z
                state.append((x, y, z, dist))
            self.trajectory.append(state)