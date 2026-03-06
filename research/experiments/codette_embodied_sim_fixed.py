import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist


class CodetteEmbodiedSimulation:
    def __init__(self, n_agents=200, n_steps=150, kappa=0.28, noise_sigma=0.035):
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.kappa = kappa
        self.noise_sigma = noise_sigma

        # Unicycle states: [x, y, theta]
        self.states = np.random.uniform(-15, 15, (n_agents, 3))
        self.states[:, 2] = np.random.uniform(-np.pi, np.pi, n_agents)

        # 3 static obstacles (real-world elements)
        self.obstacles = np.array([[8.0, 8.0], [-8.0, 8.0], [0.0, -10.0]])

        self.history = []
        self.coherence_history = []
        self.tension_history = []
        self.ethical_history = []

    def step(self):
        mean_state = np.mean(self.states, axis=0)

        for i in range(self.n_agents):
            dx = mean_state[0] - self.states[i, 0]
            dy = mean_state[1] - self.states[i, 1]
            desired_theta = np.arctan2(dy, dx)

            dtheta = self.kappa * (desired_theta - self.states[i, 2]) + np.random.normal(0, self.noise_sigma)
            self.states[i, 2] += np.clip(dtheta, -0.5, 0.5)
            self.states[i, 2] %= (2 * np.pi)

            self.states[i, 0] += np.cos(self.states[i, 2]) * 0.8
            self.states[i, 1] += np.sin(self.states[i, 2]) * 0.8

            # Obstacle repulsion + collision avoidance
            dists = cdist([self.states[i, :2]], self.obstacles)[0]
            for j, d in enumerate(dists):
                if d < 3.0:
                    repulse = (self.states[i, :2] - self.obstacles[j]) / (d + 1e-6)
                    self.states[i, :2] += repulse * (3.0 - d) * 0.4

        # Record metrics (FIXED: np.abs does not take axis= kwarg)
        mean_theta = np.mean(self.states[:, 2])
        coherence = np.mean(np.abs(np.cos(self.states[:, 2] - mean_theta)))

        mean_pos = np.mean(self.states[:, :2], axis=0)
        diff = self.states[:, :2] - mean_pos
        tension_per_agent = np.abs(diff)  # element-wise abs, no axis=
        tension = np.mean(tension_per_agent, axis=1)

        self.coherence_history.append(coherence)
        self.tension_history.append(np.mean(tension))
        self.ethical_history.append(0.9 - 0.001 * np.mean(tension))
        self.history.append(self.states.copy())

    def run(self):
        for _ in range(self.n_steps):
            self.step()
        return np.array(self.history)

    def save_data(self):
        # Last 50 steps only
        traj = np.array(self.history)[-50:]
        steps_array = np.repeat(np.arange(50), self.n_agents)  # FIXED: matches traj shape

        df = pd.DataFrame({
            'step': steps_array,
            'agent': np.tile(np.arange(self.n_agents), 50),
            'x': traj[:, :, 0].flatten(),
            'y': traj[:, :, 1].flatten(),
            'theta': traj[:, :, 2].flatten()
        })
        df.to_csv('codette_200agent_trajectories.csv', index=False)

        metrics = pd.DataFrame({
            'step': np.arange(self.n_steps),
            'phase_coherence': self.coherence_history,
            'epistemic_tension': self.tension_history,
            'ethical_alignment': self.ethical_history
        })
        metrics.to_csv('codette_200agent_metrics.csv', index=False)

    def plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(self.coherence_history, 'b-', linewidth=2)
        axs[0, 0].axhline(0.95, color='r', linestyle='--')
        axs[0, 0].set_title('Phase Coherence Gamma (200 Agents + Obstacles)')
        axs[0, 0].set_ylabel('Gamma (Eq. 11)')
        axs[0, 0].grid(True)

        ax = axs[0, 1]
        traj = np.array(self.history)
        for i in range(min(50, self.n_agents)):
            ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.3)
        ax.scatter(self.obstacles[:, 0], self.obstacles[:, 1],
                   c='red', s=100, marker='x', label='Obstacles')
        ax.set_title('Swarm Trajectories + Collision Avoidance')
        ax.legend()
        ax.grid(True)

        axs[1, 0].plot(self.tension_history, 'purple')
        axs[1, 0].set_title('Epistemic Tension Decay')
        axs[1, 0].set_ylabel('Tension')

        axs[1, 1].plot(self.ethical_history, 'green')
        axs[1, 1].set_title('AEGIS Ethical Alignment')
        axs[1, 1].set_ylabel('eta')

        plt.tight_layout()
        plt.savefig('codette_200agent_diagnostics.png', dpi=300)
        plt.close()


# === Raspberry Pi Edge Deployment (uncomment when running on hardware) ===
# from gpiozero import Robot, DistanceSensor
#
# class CodettePod:
#     def __init__(self, left_pins=(17, 18), right_pins=(22, 23), echo=24, trigger=25):
#         self.robot = Robot(left=left_pins, right=right_pins)
#         self.sensor = DistanceSensor(echo=echo, trigger=trigger)
#
#     def move_toward(self, angle, speed=0.6):
#         if self.sensor.distance < 0.3:
#             self.robot.backward(0.3)
#             return
#         if abs(angle) < 0.2:
#             self.robot.forward(speed)
#         elif angle > 0:
#             self.robot.left(speed * 0.5)
#         else:
#             self.robot.right(speed * 0.5)
#
#     def stop(self):
#         self.robot.stop()


# === RUN ===
if __name__ == "__main__":
    sim = CodetteEmbodiedSimulation(n_agents=200, n_steps=150, kappa=0.28)
    print("Running corrected 200-agent embodied RC+xi simulation...")
    sim.run()
    sim.save_data()
    sim.plot()
    print("SUCCESS — No errors.")
    print(f"Final Phase Coherence Gamma = {sim.coherence_history[-1]:.4f}")
    print(f"Final Ethical Alignment eta = {sim.ethical_history[-1]:.4f}")
    print("Files generated:")
    print("   codette_200agent_trajectories.csv")
    print("   codette_200agent_metrics.csv")
    print("   codette_200agent_diagnostics.png")
