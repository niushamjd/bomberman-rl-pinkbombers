import json
import numpy as np
import matplotlib.pyplot as plt

with open('agent_code/pink_bomber/training_scores.json') as f:
    scores = json.load(f)

scores = np.array(scores)
window = 20
rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(scores, alpha=0.3, label='Score per round')
plt.plot(range(window - 1, len(scores)), rolling_avg, label=f'{window}-round rolling average', linewidth=2)
plt.xlabel('Training round')
plt.ylabel('Score')
plt.title('Pink Bombers — Q-learning training progress (coin-heaven scenario)')
plt.legend()
plt.tight_layout()
plt.savefig('training_progress.png', dpi=150)
print("Saved training_progress.png")
