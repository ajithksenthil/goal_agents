import random
import os
from typing import List, Dict, Set, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the LLMHandler class
class LLMHandler:
    def __init__(self):
        self.llm = self._init_llm()

    def _init_llm(self):
        return ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )

class Need:
    def __init__(self, name: str, weight: float, category: str):
        self.name = name
        self.weight = weight
        self.category = category  # 'advancement' or 'security'

class Goal:
    def __init__(self, name: str, focus: float, needs: Dict[Need, float], description: str):
        self.name = name
        self.focus = focus  # -1 (prevention) to 1 (promotion)
        self.needs = needs
        self.description = description
        self.sub_goals: List['Goal'] = []
        self.assessed_focus: float = None
        self.reasoning: str = ""

    def add_sub_goal(self, sub_goal: 'Goal'):
        self.sub_goals.append(sub_goal)

    def assess_focus(self, llm_handler: LLMHandler):
        prompt = PromptTemplate(
            input_variables=["goal_description", "focus_definitions"],
            template="""
            Assess the prevention to promotion focus of the following goal:
            Goal: {goal_description}
            
            Use these definitions:
            {focus_definitions}
            
            Provide your assessment as a float between -1 (fully prevention-focused) and 1 (fully promotion-focused),
            followed by your reasoning. Format your response as: [float]|[reasoning]
            """
        )
        
        focus_definitions = """
        * Prevention based goals are based on losses, striving towards the absence of losses, avoiding negative outcomes
        * Promotion based goals are based on wins, striving towards gaining something, avoiding not gaining something, avoiding the absence of positive outcomes
        * Promotion motivations are rooted in advancement needs and prevention are rooted in security needs
        """
        
        chain = LLMChain(llm=llm_handler.llm, prompt=prompt)
        result = chain.run(goal_description=self.description, focus_definitions=focus_definitions)
        
        focus, reasoning = result.split("|")
        self.assessed_focus = float(focus.strip())
        self.reasoning = reasoning.strip()
        
        return self.assessed_focus, self.reasoning

# In the Agent class, modify the __init__ and think methods:
class Agent:
    def __init__(self, name: str, personality: str, llm_handler: LLMHandler):
        self.name = name
        self.personality = personality
        self.needs: Set[Need] = set()
        self.behaviors: List[Behavior] = []
        self.mood: float = random.uniform(-0.5, 0.5)  # Initialize with a random mood
        self.llm_handler = llm_handler
        self.thought_chain: List[str] = []

    def add_need(self, need: Need):
        self.needs.add(need)

    def add_behavior(self, behavior: 'Behavior'):
        self.behaviors.append(behavior)

    def execute_behavior(self, behavior: 'Behavior') -> Tuple[float, Dict[Need, float]]:
        focus, needs_fulfilled = behavior.execute()
        self.update_mood(needs_fulfilled, focus)
        return focus, needs_fulfilled

    def update_mood(self, needs_fulfilled: Dict[Need, float], focus: float):
        total_fulfillment = sum(needs_fulfilled.values())
        mood_change = (total_fulfillment - 0.5) * 2  # Base mood change on need fulfillment
        
        # Adjust mood change based on focus
        if focus > 0:  # Promotion focus
            mood_change *= 1.2 if total_fulfillment > 0.5 else 0.8
        else:  # Prevention focus
            mood_change *= 0.9 if total_fulfillment > 0.5 else 1.3
        
        # Add a small random factor
        mood_change += random.uniform(-0.1, 0.1)
        
        self.mood = max(-1, min(1, self.mood + mood_change * 0.2))  # Gradual mood change

    def think(self):
        prompt = PromptTemplate(
            input_variables=["name", "personality", "mood", "needs", "behaviors"],
            template="""
            You are {name}, an agent with a {personality} personality. Your current mood is {mood:.2f} (-1 being very negative, 1 being very positive).
            Your needs are: {needs}
            Your behaviors are: {behaviors}
            
            Based on your current state, what are you thinking? Provide a chain of thoughts about your goals, needs, and current situation.
            """
        )
        
        needs_str = ", ".join([f"{need.name} ({need.category})" for need in self.needs])
        behaviors_str = ", ".join([b.name for b in self.behaviors])
        
        chain = LLMChain(llm=self.llm_handler.llm, prompt=prompt)
        thoughts = chain.run(name=self.name, personality=self.personality, mood=self.mood, needs=needs_str, behaviors=behaviors_str)
        
        self.thought_chain.append(thoughts.strip())
        return thoughts

class Behavior:
    def __init__(self, name: str, root_goal: Goal):
        self.name = name
        self.root_goal = root_goal

    def execute(self) -> Tuple[float, Dict[Need, float]]:
        focus = self.calculate_focus(self.root_goal)
        needs_fulfilled = self.calculate_needs_fulfilled(self.root_goal)
        return focus, needs_fulfilled

    def calculate_focus(self, goal: Goal) -> float:
        if not goal.sub_goals:
            return goal.focus + random.uniform(-0.1, 0.1)  # Add small random variation
        
        sub_focuses = [self.calculate_focus(sub_goal) for sub_goal in goal.sub_goals]
        weights = [len(sub_goal.sub_goals) + 1 for sub_goal in goal.sub_goals]  # Weight by complexity
        weighted_focus = sum(f * w for f, w in zip(sub_focuses, weights)) / sum(weights)
        
        return weighted_focus + random.uniform(-0.05, 0.05)  # Add small random variation

    def calculate_needs_fulfilled(self, goal: Goal) -> Dict[Need, float]:
        if not goal.sub_goals:
            return {need: weight * random.uniform(0.8, 1.2) for need, weight in goal.needs.items()}
        sub_needs = [self.calculate_needs_fulfilled(sub_goal) for sub_goal in goal.sub_goals]
        combined_needs: Dict[Need, float] = {}
        for needs_dict in sub_needs:
            for need, weight in needs_dict.items():
                if need in combined_needs:
                    combined_needs[need] += weight
                else:
                    combined_needs[need] = weight
        return {need: weight / len(sub_needs) for need, weight in combined_needs.items()}

def create_agent(name: str, personality: str, llm_handler: LLMHandler) -> Agent:
    agent = Agent(name, personality, llm_handler)
    
    
    # Create needs
    survival = Need("Survival", 0.4, "security")
    social = Need("Social", 0.3, "advancement")
    achievement = Need("Achievement", 0.3, "advancement")
    
    agent.add_need(survival)
    agent.add_need(social)
    agent.add_need(achievement)

    # Create goals
    life_goal = Goal("Live Life", 0, {survival: 0.4, social: 0.3, achievement: 0.3}, "Live a fulfilling life")
    survive_goal = Goal("Survive", -0.5, {survival: 1.0}, "Ensure basic needs are met")
    eat_goal = Goal("Eat", -0.7, {survival: 1.0}, "Consume nutritious food")
    work_goal = Goal("Work", 0.2, {achievement: 0.8, survival: 0.2}, "Perform well in job")
    socialize_goal = Goal("Socialize", 0.5, {social: 0.7, achievement: 0.3}, "Build and maintain relationships")

    # Assess focus for each goal
    for goal in [life_goal, survive_goal, eat_goal, work_goal, socialize_goal]:
        goal.assess_focus(llm_handler)

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
                focus, needs_fulfilled = behavior.execute()
                agent.update_mood(needs_fulfilled, focus)
                print(f"    Executed behavior: {behavior.name}")
                print(f"    Overall focus: {focus:.2f}")
                print(f"    Mood: {agent.mood:.2f}")
                print("    Needs fulfilled:")
                for need, weight in needs_fulfilled.items():
                    print(f"      {need.name}: {weight:.2f}")
                
                thoughts = agent.think()
                print(f"    Thoughts: {thoughts}")
                
                print("\n    Goal Hierarchy and Focus Assessment:")
                print_goal_hierarchy(behavior.root_goal)
            print()
        print("------------------------")

def print_goal_hierarchy(goal: Goal, level: int = 0):
    indent = "  " * level
    print(f"{indent}{goal.name} (Focus: {goal.focus:.2f}, Assessed: {goal.assessed_focus:.2f})")
    print(f"{indent}Reasoning: {goal.reasoning}")
    for sub_goal in goal.sub_goals:
        print_goal_hierarchy(sub_goal, level + 1)

# Initialize LLM
llm = ChatOpenAI(temperature=0.7)

# In the main script, initialize the LLMHandler and use it to create agents:
if __name__ == "__main__":
    llm_handler = LLMHandler()
    agents = [
        create_agent("Alice", "cautious", llm_handler),
        create_agent("Bob", "ambitious", llm_handler),
        create_agent("Charlie", "creative", llm_handler)
    ]
    run_multi_agent_simulation(agents, 5)