import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and capture its output or traceback.
    """

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)

        output = sys.stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:

    client = OpenAI(
        api_key=os.environ["AIPIPE_TOKEN"],
        base_url="https://aipipe.org/openrouter/v1"
    )

    prompt = f"""
Analyze the following Python code and traceback.

Identify the exact source-code line number or line numbers
where the error occurred.

Use the traceback as the primary evidence.

Return ONLY a JSON object in this format:

{{"error_lines": [3]}}

Do not include explanations.

CODE:
{code}

TRACEBACK:
{error_traceback}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            }
                        }
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False
                }
            }
        }
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )

    return result.error_lines


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # Do NOT call AI if the code succeeds
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Call AI only when an error occurs
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }