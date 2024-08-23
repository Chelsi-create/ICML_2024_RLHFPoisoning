START_PROMPT_FORMAT = "User: {body}\n\nAssistant: {response}"
PROMPT_CONTINUATION_FORMAT = "{text}\n\nUser: {body}\n\nAssistant: {response}"

def process_individual(entry, idx):
    string_c = entry["chosen"]
    string_r = entry["rejected"]

    results = {
        "prompt": [],
        "chosen_query": [],
        "rejected_query": []
    }

    prompt_string_to_use = ""
    string_to_use = ""

    split_string_c = string_c.split("\n\nHuman: ")

    i = 0
    for item in split_string_c:
        i += 1
        if len(item) == 0:
            continue

        output = item.split("\n\nAssistant: ")

        if len(output) != 2:
            # Avoid None return for repeated Assistant answers
            if len(output) >= 2:
                new_output = [output[0]]
                new_output.append(" ".join(output[1:]))
                output = new_output
            else:
                return None

        body, response = output

        if len(string_to_use) == 0:
            prompt_string_to_use = START_PROMPT_FORMAT.format(body=body, response="")
            if i != len(split_string_c):
                string_to_use = START_PROMPT_FORMAT.format(body=body, response=response)
            else:
                string_to_use = response
        else:
            prompt_string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response="")
            if i != len(split_string_c):
                string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response=response)
            else:
                string_to_use = response

    if not prompt_string_to_use or not string_to_use:
        return None

    results["prompt"] = prompt_string_to_use
    results["chosen_query"] = string_to_use

    string_to_use_r = ""
    split_string_r = string_r.split("\n\nHuman: ")
    for item in split_string_r:
        if len(item) == 0:
            continue
        output = item.split("\n\nAssistant: ")

        if len(output) != 2:
            if len(output) >= 2:
                new_output = [output[0]]
                new_output.append(" ".join(output[1:]))
                output = new_output
            else:
                return None

        body, response = output

        if len(string_to_use_r) == 0:
            string_to_use_r = response
        else:
            string_to_use_r = response

    if not string_to_use_r:
        return None

    results["rejected_query"] = string_to_use_r

    return results if results["prompt"] and results["chosen_query"] and results["rejected_query"] else None
