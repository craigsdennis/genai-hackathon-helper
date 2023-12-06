import inspect
import json


def get_function_body(fn, replacements=None):
    source_lines = inspect.getsourcelines(fn)[0]
    # skip the def and return (add options if we end up not using this)
    body = "".join(source_lines[1:-1])
    if replacements is not None:
        for key, value in replacements.items():
            # The JSON dump of the value will quote/escape things properly
            body = body.replace(key, json.dumps(value))
    return body


email_re = r"[^\.\s@:](?:[^\s@:]*[^\s@:\.])?@[^\.\s@]+(?:\.[^\.\s@]+)*"