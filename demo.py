import sys
import os
import yaml
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def load_config():
    """Load agent and personality configurations."""
    with open('src/engineering_team/config/agents.yaml', 'r') as f:
        agents_config = yaml.safe_load(f)
    
    with open('src/engineering_team/config/personalities.yaml', 'r') as f:
        personalities_config = yaml.safe_load(f)
        
    return agents_config, personalities_config['personalities']

def get_demo_tasks():
    """Return a selection of demo tasks."""
    return [
        {
            'id': 'simple',
            'name': 'Simple: Find Odd Occurrences',
            'description': """
Write a Python function `find_odd_occurrences(numbers)` that accepts a list of integers 
where every number appears an even number of times except for one number which appears 
an odd number of times. The function should return that number.
            """,
            'code': """
def find_odd_occurrences(numbers):
    result = 0
    for number in numbers:
        result ^= number
    return result
"""
        },
        {
            'id': 'complex',
            'name': 'Complex: Bank Account System',
            'description': """
Implement a BankAccount class with the following requirements:
1. Initialize with an account_id (str) and initial_balance (float, default 0.0).
2. Methods:
   - deposit(amount): Adds amount. Raises ValueError if amount <= 0.
   - withdraw(amount): Subtracts amount. Raises ValueError if amount <= 0 or insufficient funds.
   - get_balance(): Returns current balance.
   - transfer(target_account, amount): Transfers money to another BankAccount instance.
3. Maintain a transaction history.
            """,
            'code': """
class BankAccount:
    def __init__(self, account_id, initial_balance=0.0):
        self.account_id = account_id
        self.balance = initial_balance
        self.transactions = []

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount
        self.transactions.append(('deposit', amount))

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if self.balance < amount:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        self.transactions.append(('withdraw', amount))

    def get_balance(self):
        return self.balance

    def transfer(self, target, amount):
        self.withdraw(amount)
        target.deposit(amount)
        self.transactions.append(('transfer_out', amount))
        target.transactions.append(('transfer_in', amount))
"""
        }
    ]

def run_demo():
    parser = argparse.ArgumentParser(description='AI QA Agent Personality Demo')
    parser.add_argument('--personality', type=int, help='Personality index (1-4)')
    parser.add_argument('--task', type=int, help='Task index (1-2)')
    args = parser.parse_args()

    print("\n=== AI QA Agent Personality Demo ===\n")
    
    agents_config, personalities = load_config()
    
    # 1. Select Personality
    print("Select a Personality Profile:")
    options = [p for p in personalities if p['id'] in ['neutral', 'neuroticism_high', 'extraversion_high', 'openness_high']]
    for i, p in enumerate(options):
        print(f"{i+1}. {p['label']}")
    
    if args.personality:
        p_idx = args.personality - 1
        if 0 <= p_idx < len(options):
            selected_personality = options[p_idx]
        else:
            print("Invalid selection from args. Defaulting to Neutral.")
            selected_personality = options[0]
    else:
        try:
            p_idx = int(input("\nEnter choice (1-4): ")) - 1
            selected_personality = options[p_idx]
        except (ValueError, IndexError):
            print("Invalid selection. Defaulting to Neutral.")
            selected_personality = options[0]
        
    print(f"\nSelected: {selected_personality['label']}")
    print(f"Description: {selected_personality['description']}")

    # 2. Select Task
    print("\nSelect a Task:")
    tasks = get_demo_tasks()
    for i, t in enumerate(tasks):
        print(f"{i+1}. {t['name']}")
        
    if args.task:
        t_idx = args.task - 1
        if 0 <= t_idx < len(tasks):
            selected_task = tasks[t_idx]
        else:
            print("Invalid selection from args. Defaulting to Simple.")
            selected_task = tasks[0]
    else:
        try:
            t_idx = int(input("\nEnter choice (1-2): ")) - 1
            selected_task = tasks[t_idx]
        except (ValueError, IndexError):
            print("Invalid selection. Defaulting to Simple.")
            selected_task = tasks[0]

    print(f"\nSelected Task: {selected_task['name']}")
    
    # 3. Prepare Environment
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    
    module_path = output_dir / 'solution.py'
    with open(module_path, 'w') as f:
        f.write(selected_task['code'])
        
    print(f"\nRunning Agent... (This may take 30-60 seconds)")
    
    # 4. Configure Agent
    inputs = {
        'personality': f"{selected_personality['label']}: {selected_personality['description']}",
        'module_name': 'solution.py'
    }
    
    # Format agent config strings
    role = agents_config['qa']['role'].format(**inputs)
    goal = agents_config['qa']['goal'].format(**inputs)
    backstory = agents_config['qa']['backstory'].format(**inputs)
    
    qa_agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=agents_config['qa']['llm'],
        verbose=False
    )
    
    test_task = Task(
        description=f"""
        You are tasked with writing a comprehensive `pytest` test suite for the Python module located at: {module_path.absolute()}.
        
        The module implements the following requirements:
        {selected_task['description']}
        
        Your output must be a single valid Python file containing the test suite.
        Do not use markdown formatting (```python). Just return the raw code.
        """,
        expected_output="A valid python file containing pytest unit tests.",
        agent=qa_agent
    )
    
    # 5. Run Crew
    crew = Crew(
        agents=[qa_agent],
        tasks=[test_task],
        process=Process.sequential,
        verbose=False
    )
    
    start_time = time.time()
    result = crew.kickoff()
    duration = time.time() - start_time
    
    # 6. Show Results
    output_file = output_dir / f"test_{selected_task['id']}_{selected_personality['id']}.py"
    with open(output_file, 'w') as f:
        f.write(str(result))
        
    print("\n" + "="*50)
    print(f"GENERATION COMPLETE in {duration:.2f} seconds")
    print("="*50)
    print(f"\nOutput saved to: {output_file}")
    print("\n--- Generated Code Preview ---\n")
    
    # Print first 20 lines
    lines = str(result).split('\n')
    for line in lines[:20]:
        print(line)
    if len(lines) > 20:
        print("... (rest of file omitted) ...")

if __name__ == "__main__":
    run_demo()
