#!/usr/bin/env python3
import os
import re
import json
import traceback
from typing import Any

from utils.fetch import fetch

# TODO: replace with appropriate LLM service URL
base_url = f"TODO:REPLACE"

project = "semantic-similarity"
user_id = os.getenv('LLM_API_USER_ID')

def call_llm(call:str, context:dict, model:str, verbose:bool=False, return_json:bool=True) -> dict:
    """
      {call}     - name of the prompt template to call
      {context}  - context for the call
      {model}    - model to use for the call
    """
    call_url = f"{base_url}/{call}"
    context = normalize_context(context)

    if verbose:
        print(f"Test: {call}\nContext: {context}\nModel: {model}")

    body = {
        "context": context,
        "user_id": user_id,
        "project": project,
        "model": model
    }

    try:
        api_response = fetch({"url": call_url, "body": body}, complete_response=True)
        response, used_model = extract_response(api_response, return_json=return_json)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return {}

    if not used_model:
        response = f"error when calling model '{model}': {response}"
        used_model = model

    if verbose:
        print(f"Response: {response}\n\n")
    return {"call": call, "context": context, "model": used_model, "response": response}

def is_fine_response(r:Any, call_url:str) -> bool:
    success_criteria = [
        ("extract-predicates-from-abstract", list),
    ]
    return any([call in call_url and isinstance(r.get("response"), response_type) for call, response_type in success_criteria])

def normalize_context(context:dict) -> dict:
    """ Convert all non-string values to strings """
    result = context 
    for key, value in result.items():
        if type(value) in [int, float]:
            result[key] = str(value)
        elif type(value) in [dict, list]:
            result[key] = json.dumps(value)
    return result

def extract_response(api_response:dict, return_json:bool=True) -> tuple[str|list, str]:
    used_model = None
    json_pattern = re.compile(".*```json([^`]+)```.*")

    if not api_response:
        response = "empty response"

    elif api_response["status"] == "ok" and api_response.get("result"):
        api_result = api_response.get("result")
        used_model = api_result.get("model")
        response = api_result.get("response", "")

        if not return_json:
            return response, used_model
        response = re.sub(json_pattern, "\\1", response).strip()

        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            # one more attempt to catch json list/dict in case {json_pattern} did not do the job
            for left_tag, right_tag in [("[", "]"), ("{", "}")]:
                start = response.find(left_tag)
                end = response.rfind(right_tag) + 1
                if start >= 0 and end > start:
                    response = response[start:end]
                    break

                try:
                    response = json.loads(response)
                except json.JSONDecodeError as e:
                    pass
    else:
        response = api_response.get('reason', 'unknown reason')

    return response, used_model