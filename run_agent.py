#!/usr/bin/env python3
import re
import json
import copy
import traceback
from typing import Any

from utils import Embedder, call_llm, set_encoder, SmartList

embedder = Embedder(model="Alibaba-NLP/gte-large-en-v1.5")

call = "semantic-agent"
model = "gpt-4.1"

obs_in = "<OBSERVATION>\n"
obs_out = "\n</OBSERVATION>\n\n"
RED = '\033[91m'
GREEN = '\033[32m'
GRAY = '\033[90m'
ENDC = '\033[0m'

### Initialization of state management

# you can change the initial input index to resume from a specific point
initial_input_index = 0
#initial_input_index = 177

state = {}
# full history of observations and actions
history = []
# part of the history that gets passed to the LLM
llm_context = []

initial_state = ""

def init_state():
    """
    Initialize the state for the agent.
    """
    global state, initial_state
    state = {
        'input': [],
        'output': SmartList()
    }

    with open("out/polished_clusters.json", "r") as f:
        state['input'] = json.load(f)

    state['input_index'] = initial_input_index
    initial_state = f"```input_index = {state['input_index']}```"
    if initial_input_index > 0:
        with open("out/agent_output.json", "r") as f:
            state['output'] = json.load(f)[:4]


### Custom utilities for agent
def init_embedder(clusters: list[dict]|None = None) -> Embedder:
    """
    Initialize the embedder for processing predicates.
    """
    embedder.build_index(clusters=clusters, use_rationale=True, rebuild=True, export=False)
    return embedder

def find(query:list[dict], k:int=3) -> str|None:
    key = ' '.join([subset['rationale'] for subset in query])

    embedder = init_embedder(state['output'])
    matching_indices = embedder.search(key, n=k, return_index=True)

    print(RED + f"SYSTEM REMARK: Found matching indices {matching_indices} for query: {key} (output length {len(state['output'])})." + ENDC)
    result = ""
    for i in matching_indices:
        result += obs_in + f"# output[{i}]\n" + json.dumps(state['output'][i], indent=4) + "\n" + obs_out
    return result or None

### Agent business logic

def observe(variable:Any) -> list[str]:
    result = ""
    query = variable
    if type(query) in [dict, list]:
        result = obs_in + json.dumps(query, indent=4) + obs_out
    elif type(query) is str:
        result = query.strip()
    else:
        result = obs_in + str(query) + obs_out
    print(GRAY + result + ENDC)

    history.append(result)
    llm_context.append(result)
    return llm_context

def evaluate_code(code:str) -> tuple[bool, str]:
    """
    Evaluate the provided code and return the result.
    """
    llm_context.append(code)
    history.append(code)
    try:
        # Create a deep copy of the state to avoid modifying the original state
        local_vars = copy.deepcopy(state)
        exec(code, {"observe": observe, "find": find}, local_vars)

        # ignore any updates to the input
        del local_vars['input']
        state.update(local_vars)

        status, code_eval_result = (True, obs_in + f"Code evaluation was successful" + obs_out)
    except Exception as e:
        status, code_eval_result = (False, obs_in + f"Error evaluating code: {e}" + obs_out)
        print(RED + f"Error: {e}" + ENDC)
        traceback.print_exc()

    llm_context.append(code_eval_result)
    history.append(code_eval_result)
    return (status, code_eval_result)

def reset_context(response:str) -> None:
    """
    Reset the LLM context depending on the response.
    """
    global llm_context
    reset_regex = re.compile(r"(\s*input_index\s*\+=\s*\d+)")

    if response and reset_regex.search(response):
        llm_context = llm_context[-2:]
        # print in red color
        print(RED + "SYSTEM REMARK: Context reset due to input index change." + ENDC)

def main(max_fail_count:int=3, human_in_the_loop:bool=False):
    """
    Main function to run the agent.
    """
    init_state()
    fail_count = 0

    # event loop to keep the agent running
    print("Agent is running. Press Ctrl+C to stop.")
    while True:
        try:
            # call LLM
            context = '\n'.join(llm_context) or "[]"           
            response = call_llm(call=call, context={"history": context, "initial_state": initial_state}, model=model, return_json=False)

            print(GREEN + f"<LLM>\n{response['response']}\n</LLM>\n" + ENDC)            

            response = response.get("response", "None")
            # evaluate the response
            status, code_eval_result = evaluate_code(response)
            if not status:
                # try again with the current context
                fail_count += 1
                if fail_count >= max_fail_count:
                    print(RED + f"Maximum failure count reached ({max_fail_count}). Exiting." + ENDC)
                    break
                continue
            
            fail_count = 0
            if state['input_index'] >= len(state['input']):
                print(GREEN + "All inputs processed. Exiting." + ENDC)
                break

            print(code_eval_result)

            # conditional reset of context (trigger depends on the response)
            reset_context(response)

            if human_in_the_loop:
                # wait for input to continue
                user_input = input("Enter a command (or 'exit' to quit): ").strip()
        except KeyboardInterrupt:
            print("\nExiting the agent.")
            break
        except Exception as e:
            raise e

    try:
        # Save the output to a file
        with open("out/agent_output.json", "w") as f:
            json.dump(state['output'], f, indent=4, default=set_encoder)
    except Exception as e:
        print(RED + f"Error saving output: {e}" + ENDC)
        traceback.print_exc()

    try:
        with open("out/agent_history.json", "w") as f:
            json.dump(history, f, indent=4, default=set_encoder)
    except Exception as e:
        print(RED + f"Error saving history: {e}" + ENDC)
        traceback.print_exc()

    try:
        # support serialization of sets to lists
        with open("out/agent_debug.json", "w") as f:
            json.dump({**state, "history": history, "llm_context": llm_context}, f, indent=4, default=set_encoder)
    except Exception as e:
        print(RED + f"Error saving debug information: {e}" + ENDC)
        traceback.print_exc()


    print(f"Processed {len(state['output'])} items and saved to 'out/agent_output.json'")

if __name__ == "__main__":
    main()