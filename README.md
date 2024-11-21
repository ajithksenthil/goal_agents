# Goal-Oriented Agent Simulation

A hierarchical goal-based agent simulation system that models agent behavior through needs, goals, and personality traits. The project includes multiple implementations, from basic agent simulations to advanced versions with LLM integration for dynamic behavior generation.

## Project Structure

```
goal_agents/
├── goal_prev_prom_agent.py        # Basic single-agent implementation
├── goal_prev_prom_multiagent.py   # Basic multi-agent implementation
├── intelligent_goal_agents.py      # Advanced agent with LLM for goal creation
├── intelligent_goal_thinking_agents.py  # Advanced agent with memory and thinking
├── requirements.txt               # Project dependencies
├── .env                          # Environment variables (for OpenAI API key)
└── README.md                     # This file
```

## Features

### Basic Implementation (`goal_prev_prom_agent.py`, `goal_prev_prom_multiagent.py`)
- Core goal-oriented agent architecture
- Hierarchical goal system
- Need-based behavior modeling
- Prevention-Promotion focus system
- Multi-agent simulation with different personalities

### Advanced Implementation (`intelligent_goal_agents.py`)
- LLM-powered dynamic goal creation
- Personality-influenced goal assessment
- Enhanced mood system
- Detailed goal hierarchy visualization
- Goal description and reasoning

### Intelligent Implementation (`intelligent_goal_thinking_agents.py`)
- Memory system for tracking events and experiences
- Dynamic thought generation through LLM
- Advanced mood regulation
- Event-based memory system
- Goal assessment with LLM-based reasoning

## Installation

1. Create a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Single Agent Simulation
```python
from goal_prev_prom_agent import create_agent_with_behaviors, run_simulation

agent = create_agent_with_behaviors()
run_simulation(agent, 5)  # Run for 5 days
```

### Basic Multi-Agent Simulation
```python
from goal_prev_prom_multiagent import create_agent, run_multi_agent_simulation

agents = [
    create_agent("Alice", "cautious"),
    create_agent("Bob", "ambitious"),
    create_agent("Charlie", "balanced")
]
run_multi_agent_simulation(agents, 5)
```

### Advanced Agent Simulation with LLM
```python
from intelligent_goal_agents import create_agent, run_multi_agent_simulation, LLMHandler

llm_handler = LLMHandler()
agents = [
    create_agent("Alice", "cautious", llm_handler),
    create_agent("Bob", "ambitious", llm_handler),
    create_agent("Charlie", "creative", llm_handler)
]
run_multi_agent_simulation(agents, 5)
```

### Intelligent Agent Simulation with Memory and Thinking
```python
from intelligent_goal_thinking_agents import create_agent, run_multi_agent_simulation, LLMHandler

llm_handler = LLMHandler()
agents = [
    create_agent("Alice", "cautious", llm_handler),
    create_agent("Bob", "ambitious", llm_handler),
    create_agent("Charlie", "creative", llm_handler)
]
run_multi_agent_simulation(agents, 5)
```

## Core Components

### Need
- Represents fundamental requirements
- Includes weight and category (security/advancement)
- Basic needs: Survival, Social, Achievement

### Goal
- Hierarchical structure with sub-goals
- Focus value ranging from -1 (prevention) to 1 (promotion)
- Associated needs with weights
- Description and reasoning (in advanced versions)

### Behavior
- Represents agent actions
- Connected to root goals
- Calculates need fulfillment and focus

### Agent
- Contains needs, behaviors, and state
- Personality traits influence behavior
- Memory system (in intelligent version)
- Dynamic thought generation (in intelligent version)

## Dependencies
```
langchain
langchain-openai
python-dotenv
typing-extensions
openai
```

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Notes

- The basic implementation can run without OpenAI API key
- Advanced and intelligent implementations require an OpenAI API key
- Different personality types (cautious, ambitious, creative, balanced) affect goal orientation
- Memory system in the intelligent implementation maintains a history of events and mood changes
- LLM integration provides dynamic goal creation and thought generation