import random
from typing import List, Dict, Set, Tuple

class Need:
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

class Goal:
    def __init__(self, name: str, focus: float, needs: Dict[Need, float]):
        self.name = name
        self.base_focus = focus  # -1 to 1, prevention to promotion
        self.needs = needs  # Dict of Need to weight
        self.sub_goals: List['Goal'] = []

    def add_sub_goal(self, sub_goal: 'Goal'):
        self.sub_goals.append(sub_goal)

    def calculate_focus(self) -> float:
        if not self.sub_goals:
            return self.base_focus + random.uniform(-0.1, 0.1)  # Add small random variation
        return sum(sub_goal.calculate_focus() for sub_goal in self.sub_goals) / len(self.sub_goals)

class Behavior:
    def __init__(self, name: str, root_goal: Goal):
        self.name = name
        self.root_goal = root_goal

class Agent:
    def __init__(self, name: str):
        self.name = name
        self.needs: Set[Need] = set()
        self.behaviors: List[Behavior] = []

    def add_need(self, need: Need):
        self.needs.add(need)

    def add_behavior(self, behavior: Behavior):
        self.behaviors.append(behavior)

    def execute_behavior(self, behavior: Behavior) -> Tuple[float, Dict[Need, float]]:
        focus = behavior.root_goal.calculate_focus()
        needs_fulfilled = self.calculate_needs_fulfilled(behavior.root_goal)
        return focus, self.normalize_needs(needs_fulfilled)

    def calculate_needs_fulfilled(self, goal: Goal) -> Dict[Need, float]:
        if not goal.sub_goals:
            return {need: weight * random.uniform(0.8, 1.2) for need, weight in goal.needs.items()}  # Add variation
        sub_goal_needs = [self.calculate_needs_fulfilled(sub_goal) for sub_goal in goal.sub_goals]
        combined_needs: Dict[Need, float] = {}
        for needs_dict in sub_goal_needs:
            for need, weight in needs_dict.items():
                if need in combined_needs:
                    combined_needs[need] += weight
                else:
                    combined_needs[need] = weight
        return combined_needs

    def normalize_needs(self, needs: Dict[Need, float]) -> Dict[Need, float]:
        total = sum(needs.values())
        return {need: weight / total for need, weight in needs.items()}

def create_agent(name: str, personality: str) -> Agent:
    agent = Agent(name)
    
    # Create needs
    survival = Need("Survival", 0.4)
    social = Need("Social", 0.3)
    achievement = Need("Achievement", 0.3)
    
    agent.add_need(survival)
    agent.add_need(social)
    agent.add_need(achievement)

    # Adjust focus based on personality
    if personality == "cautious":
        eat_focus, work_focus, socialize_focus = -0.7, 0.0, 0.5
    elif personality == "ambitious":
        eat_focus, work_focus, socialize_focus = -0.3, 0.6, 0.4
    else:  # balanced
        eat_focus, work_focus, socialize_focus = -0.5, 0.2, 0.7

    # Create goals and behaviors
    eat_goal = Goal("Eat", eat_focus, {survival: 1.0})
    socialize_goal = Goal("Socialize", socialize_focus, {social: 0.7, achievement: 0.3})
    work_goal = Goal("Work", work_focus, {achievement: 0.8, survival: 0.2})

    survive_goal = Goal("Survive", 0, {survival: 1.0})
    survive_goal.add_sub_goal(eat_goal)

    life_goal = Goal("Live Life", 0, {survival: 0.4, social: 0.3, achievement: 0.3})
    life_goal.add_sub_goal(survive_goal)
    life_goal.add_sub_goal(socialize_goal)
    life_goal.add_sub_goal(work_goal)

    behavior = Behavior("Daily Routine", life_goal)
    agent.add_behavior(behavior)

    return agent

def run_multi_agent_simulation(agents: List[Agent], num_days: int):
    for day in range(1, num_days + 1):
        print(f"Day {day}:")
        for agent in agents:
            print(f"  Agent: {agent.name}")
            for behavior in agent.behaviors:
                focus, needs_fulfilled = agent.execute_behavior(behavior)
                print(f"    Executed behavior: {behavior.name}")
                print(f"    Overall focus: {focus:.2f}")
                print("    Needs fulfilled:")
                for need, weight in needs_fulfilled.items():
                    print(f"      {need.name}: {weight:.2f}")
            print()
        print("------------------------")

# Run the multi-agent simulation
if __name__ == "__main__":
    agents = [
        create_agent("Alice", "cautious"),
        create_agent("Bob", "ambitious"),
        create_agent("Charlie", "balanced")
    ]
    run_multi_agent_simulation(agents, 5)