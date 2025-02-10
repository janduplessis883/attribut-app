import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
from datetime import datetime, timedelta

def generate_random_tasks(num_tasks=5):
    """Generate synthetic tasks with realistic attributes."""
    tasks = []
    priority_weights = {"High": 1.0, "Medium": 0.5, "Low": 0.2}

    for i in range(num_tasks):
        # Generate random deadline within next 7 days
        deadline = datetime.now() + timedelta(hours=random.randint(1, 168))
        tasks.append({
            "Task ID": f"TASK-{i+1:03d}",
            "Priority": random.choice(list(priority_weights.keys())),
            "Duration (hrs)": round(random.uniform(0.15, 3.0), 2),
            "Deadline": deadline.strftime("%Y-%m-%d %H:%M"),
            "Dependency": random.choice([True, False]),
            "Energy Level": random.choice(["Low", "Medium", "High"])
        })
    return tasks

def create_task_tensor(tasks, current_time):
    pass

class TaskScheduler(nn.Module):
    def __init__(self, input_size=50, hidden_size=64):
        """
        input_size=50 => 1 (time) + 10 tasks×4 + 7 day-mask + 2 work-hours
        hidden_size=64 => Arbitrary hidden layer size
        """
        super().__init__()
        # Each task is 4 features, plus 10 "context" features => total 14
        self.net = nn.Sequential(
            nn.Linear(14, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        """
        state shape: [batch_size, 50]
          - index 0: normalized_time
          - next 40 (10 tasks × 4 features)
          - last 9 (workdays + work hours)
        """
        batch_size = state.size(0)

        # Extract context
        normalized_time = state[:, 0].unsqueeze(1)  # [batch, 1]
        workdays = state[:, -9:-2]                  # [batch, 7]
        work_hours = state[:, -2:]                  # [batch, 2]
        context = torch.cat([normalized_time, workdays, work_hours], dim=1)  # [batch, 10]

        # Extract tasks
        task_feats = state[:, 1:-9]                 # [batch, 40]
        num_tasks = task_feats.shape[1] // 4
        tasks = task_feats.view(batch_size, num_tasks, 4)  # [batch, 10, 4]

        # Score each task
        scores = []
        for i in range(num_tasks):
            task = tasks[:, i, :]  # [batch, 4]
            combined = torch.cat([task, context], dim=1)  # [batch, 14]
            scores.append(self.net(combined))  # [batch, 1]

        scores = torch.stack(scores, dim=1).squeeze(-1)  # [batch, 10]
        return torch.softmax(scores, dim=1)              # Probability distribution

class PPOTrainer:
    def __init__(self, model, lr=1e-4, gamma=0.99, clip_epsilon=0.2, db_client=None):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = []
        self.db_client = db_client  # optional DB client

    def compute_reward(self, selected_task, current_time):
        """
        A basic example of a composite reward function, extracting 4 features:
          selected_task = [priority, deadline_urgency, duration_norm, dependency_flag]
        Feel free to customize this logic.
        """
        priority = selected_task[0].item()          # in {1.0, 0.5, 0.2}
        deadline_urgency = selected_task[1].item()  # in [0..1]
        duration = selected_task[2].item() * 8.0    # denormalize
        dep_flag = selected_task[3].item()          # 1.0 or 0.0

        # Example: bigger reward for higher priority, more urgent tasks, and shorter durations
        # also small penalty if there's a dependency
        # This is just a placeholder formula
        reward = 2.0 * priority + 1.0 * deadline_urgency - 0.2 * duration - 0.5 * dep_flag

        # Optional random user feedback factor:
        if random.random() < 0.7:
            reward += 1.0  # user liked it 70% of the time
        else:
            reward -= 2.0

        return reward

    def store_transition(self, state, action, old_prob, reward, task_id):
        self.memory.append((state, action, old_prob, reward, task_id))

    def update_policy(self):
        if not self.memory:
            return None

        # Unpack memory
        states, actions, old_probs, rewards, task_ids = zip(*self.memory)
        states = torch.cat(states)   # shape [batch_size, 50]
        actions = torch.tensor(actions)
        old_probs = torch.cat(old_probs)
        rewards = torch.tensor(rewards)

        # Discount rewards
        discounted_rewards = []
        running_reward = 0
        for r in reversed(rewards):
            running_reward = r + self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        discounted_rewards = torch.tensor(discounted_rewards)

        # Normalize
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
                             (discounted_rewards.std() + 1e-8)

        # Get new probabilities
        new_probs = self.model(states).gather(1, actions.unsqueeze(1))  # shape [batch_size, 1]
        ratio = (new_probs / old_probs).squeeze()

        # PPO loss
        surr1 = ratio * discounted_rewards
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * discounted_rewards
        loss = -torch.min(surr1, surr2).mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log transitions if db_client is set
        if self.db_client is not None:
            self.log_to_db(states, actions, rewards, discounted_rewards, loss.item(), task_ids)

        # Clear memory
        self.memory = []
        return loss.item()

    def log_to_db(self, states, actions, rewards, discounted_rewards, loss_value, task_ids):
        """Example: Write step-by-step info to your database (e.g. Supabase)."""
        for i in range(len(actions)):
            # Convert the i-th state to a list of floats
            state_list = states[i].tolist()  # e.g. [0.12, 0.87, ...]

            row = {
                "task_id": task_ids[i] if task_ids[i] else "Unknown",
                "action": int(actions[i].item()),
                "reward": float(rewards[i].item()),
                "discounted_reward": float(discounted_rewards[i].item()),
                "loss": loss_value,
                "timestamp": datetime.now().isoformat(),
                "state_embedding": json.dumps(state_list)  # store as JSON string
            }
            self.db_client.table("rl_logs").insert(row).execute()

def live_training_cycle(num_epochs=1000, batch_size=32):
    """
    An example training loop using PPO.
    This function is optional if you're controlling training from your Streamlit app.
    """
    model = TaskScheduler()  # fresh model
    trainer = PPOTrainer(model)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    for epoch in range(num_epochs):
        for _ in range(batch_size):
            tasks = generate_random_tasks(num_tasks=5)  # up to 10
            state = create_task_tensor(tasks, current_time)  # [1, 50]

            with torch.no_grad():
                probs = model(state)  # shape [1, 10]

            action = torch.multinomial(probs, 1).item()

            # Identify the chosen task features in the state
            selected_task = state[0, 1 + action*4 : 1 + (action+1)*4]
            reward = trainer.compute_reward(selected_task, current_time)

            # Store the transition
            trainer.store_transition(
                state=state,
                action=action,
                old_prob=probs[:, action],
                reward=reward,
                task_id=f"TASK-action-{action}"
            )

        # PPO update
        loss = trainer.update_policy()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Loss: {loss:.4f}")

def evaluate_tasks(tasks, model, current_time=None):
    """
    Evaluate a list of tasks with the model and return them ranked by descending probability.
    Also returns the raw probability tensor.
    """
    if current_time is None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    state_tensor = create_task_tensor(tasks, current_time)
    model.eval()
    with torch.no_grad():
        probs = model(state_tensor)  # shape [1, 10]

    sorted_indices = torch.argsort(probs[0], descending=True)
    ranked_tasks = [tasks[i] for i in sorted_indices if i < len(tasks)]

    return ranked_tasks, probs

if __name__ == "__main__":
    # Example standalone run
    live_training_cycle(num_epochs=100, batch_size=8)
