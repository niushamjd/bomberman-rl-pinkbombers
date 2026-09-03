# Bomberman RL — Pink Bombers

## Team
- Ipek Uzun
- Buse Erkiraz
- Niyousha Mojoudi

## Repository Structure

- `agent_code/pink_bomber/` — our submitted agent (best-performing model)
- `agent_code/` — other agent variants and the course-provided template/rule-based/random agents
- `environment.py`, `main.py`, `settings.py` — the game framework (provided by the course, [ukoethe/bomberman_rl](https://github.com/ukoethe/bomberman_rl))
- `replays/`, `logs/` — generated during training/testing, not core project code

## Running the Agent

\`\`\`bash
python main.py play --my-agent pink_bomber
\`\`\`

## Training

\`\`\`bash
python main.py play --no-gui --agents pink_bomber --train 1 --scenario coin-heaven
\`\`\`

(See the project report for full details on training scenarios, reward shaping, and experimental results.)
