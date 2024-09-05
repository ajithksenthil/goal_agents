import random
import os
from typing import List, Dict, Set, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
# Load environment variables
load_dotenv()

class LLMHandler:
    def __init__(self):
        self.llm = self._init_llm()

    def _init_llm(self):
        return ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )

llm_handler = LLMHandler()

class Need:
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

class Goal:
    def __init__(self, name: str, focus: float, needs: Dict[Need, float], description: str):
        self.name = name
        self.base_focus = focus  # -1 to 1, prevention to promotion
        self.needs = needs  # Dict of Need to weight
        self.sub_goals: List['Goal'] = []
        self.description = description

    def add_sub_goal(self, sub_goal: 'Goal'):
        self.sub_goals.append(sub_goal)

    def calculate_focus(self) -> float:
        if not self.sub_goals:
            return self.base_focus + random.uniform(-0.1, 0.1)  # Add small random variation
        return sum(sub_goal.calculate_focus() for sub_goal in self.sub_goals) / len(self.sub_goals)

    def __str__(self):
        return f"{self.name} (Focus: {self.base_focus:.2f})"

    def print_hierarchy(self, level=0):
        print("  " * level + str(self))
        for sub_goal in self.sub_goals:
            sub_goal.print_hierarchy(level + 1)

class Behavior:
    def __init__(self, name: str, root_goal: Goal):
        self.name = name
        self.root_goal = root_goal

class Agent:
    def __init__(self, name: str, personality: str):
        self.name = name
        self.personality = personality
        self.needs: Set[Need] = set()
        self.behaviors: List[Behavior] = []
        self.mood: float = 0.0  # -1 to 1, negative to positive

    def add_need(self, need: Need):
        self.needs.add(need)

    def add_behavior(self, behavior: Behavior):
        self.behaviors.append(behavior)

    def execute_behavior(self, behavior: Behavior) -> Tuple[float, Dict[Need, float]]:
        focus = behavior.root_goal.calculate_focus()
        needs_fulfilled = self.calculate_needs_fulfilled(behavior.root_goal)
        self.update_mood(needs_fulfilled)
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

    def update_mood(self, needs_fulfilled: Dict[Need, float]):
        total_fulfillment = sum(needs_fulfilled.values())
        self.mood = (self.mood + (total_fulfillment - 0.5) * 2) / 2  # Update mood based on need fulfillment
        self.mood = max(-1, min(1, self.mood))  # Ensure mood stays between -1 and 1

def create_dynamic_goal(name: str, agent_personality: str) -> Goal:
    prompt = PromptTemplate(
        input_variables=["name", "personality"],
        template="Create a goal named {name} for an agent with {personality} personality. Provide a brief description, a focus value between -1 (prevention) and 1 (promotion), and weights for survival, social, and achievement needs (should sum to 1). Format: [Name]|Description: [description]|Focus: [value]|Survival: [value], Social: [value], Achievement: [value]"
    )
    chain = LLMChain(llm=llm_handler.llm, prompt=prompt)
    result = chain.run(name=name, personality=agent_personality)
    
    # Split the result and handle potential errors
    parts = result.split("|")
    if len(parts) < 3:
        raise ValueError(f"Unexpected format in LLM output: {result}")
    
    # Extract description (it might be in the second or third part)
    description_part = next((part for part in parts if part.startswith("Description:")), None)
    if description_part:
        description = description_part.split("Description:")[1].strip()
    else:
        description = parts[0].strip()  # Fallback to using the first part as description
    
    # Extract focus value
    focus_part = next((part for part in parts if "Focus:" in part), None)
    if not focus_part:
        raise ValueError(f"Could not find Focus in: {result}")
    focus_match = re.search(r'Focus:\s*([-+]?\d*\.?\d+)', focus_part)
    if not focus_match:
        raise ValueError(f"Could not extract focus value from: {focus_part}")
    focus = float(focus_match.group(1))
    
    # Extract need weights
    needs_part = parts[-1]  # Assume the last part contains the needs
    need_weights = {}
    for need in ['Survival', 'Social', 'Achievement']:
        match = re.search(fr'{need}:\s*([-+]?\d*\.?\d+)', needs_part)
        if not match:
            raise ValueError(f"Could not extract {need} weight from: {needs_part}")
        need_weights[need] = float(match.group(1))
    
    # Normalize weights if they don't sum to 1
    total_weight = sum(need_weights.values())
    if abs(total_weight - 1.0) > 0.01:  # Allow for small floating-point discrepancies
        need_weights = {k: v / total_weight for k, v in need_weights.items()}
    
    return Goal(name, focus, {
        Need("Survival", 0.4): need_weights['Survival'],
        Need("Social", 0.3): need_weights['Social'],
        Need("Achievement", 0.3): need_weights['Achievement']
    }, description)



# Debugging function
def debug_goal_creation(name: str, personality: str):
    try:
        goal = create_dynamic_goal(name, personality)
        print(f"Successfully created goal: {goal.name}")
        print(f"Description: {goal.description}")
        print(f"Focus: {goal.base_focus}")
        for need, weight in goal.needs.items():
            print(f"{need.name}: {weight}")
    except ValueError as e:
        print(f"Error creating goal '{name}': {e}")
        print("Raw LLM output:")
        prompt = PromptTemplate(
            input_variables=["name", "personality"],
            template="Create a goal named {name} for an agent with {personality} personality. Provide a brief description, a focus value between -1 (prevention) and 1 (promotion), and weights for survival, social, and achievement needs (should sum to 1). Format: [Name]|Description: [description]|Focus: [value]|Survival: [value], Social: [value], Achievement: [value]"
        )
        chain = LLMChain(llm=llm_handler.llm, prompt=prompt)
        result = chain.run(name=name, personality=personality)
        print(result)

def create_agent(name: str, personality: str) -> Agent:
    agent = Agent(name, personality)
    
    # Create needs
    survival = Need("Survival", 0.4)
    social = Need("Social", 0.3)
    achievement = Need("Achievement", 0.3)
    
    agent.add_need(survival)
    agent.add_need(social)
    agent.add_need(achievement)

    # Create dynamic goals
    life_goal = create_dynamic_goal("Live Life", personality)
    survive_goal = create_dynamic_goal("Survive", personality)
    eat_goal = create_dynamic_goal("Eat", personality)
    work_goal = create_dynamic_goal("Work", personality)
    socialize_goal = create_dynamic_goal("Socialize", personality)

    # Build goal hierarchy
    survive_goal.add_sub_goal(eat_goal)
    life_goal.add_sub_goal(survive_goal)
    life_goal.add_sub_goal(work_goal)
    life_goal.add_sub_goal(socialize_goal)

    behavior = Behavior("Daily Routine", life_goal)
    agent.add_behavior(behavior)

    return agent

def run_multi_agent_simulation(agents: List[Agent], num_days: int):
    for day in range(1, num_days + 1):
        print(f"Day {day}:")
        for agent in agents:
            print(f"  Agent: {agent.name} (Personality: {agent.personality})")
            for behavior in agent.behaviors:
                focus, needs_fulfilled = agent.execute_behavior(behavior)
                print(f"    Executed behavior: {behavior.name}")
                print(f"    Overall focus: {focus:.2f}")
                print(f"    Mood: {agent.mood:.2f}")
                print("    Needs fulfilled:")
                for need, weight in needs_fulfilled.items():
                    print(f"      {need.name}: {weight:.2f}")
                print("\n    Goal Hierarchy:")
                behavior.root_goal.print_hierarchy()
            print()
        print("------------------------")

# Run the multi-agent simulation
if __name__ == "__main__":
    debug_goal_creation("Live Life", "cautious")
    debug_goal_creation("Survive", "cautious")
    debug_goal_creation("Eat", "cautious")
    debug_goal_creation("Work", "cautious")
    debug_goal_creation("Socialize", "cautious")
    agents = [
        create_agent("Alice", "cautious"),
        create_agent("Bob", "ambitious"),
        create_agent("Charlie", "creative")
    ]
    run_multi_agent_simulation(agents, 5)