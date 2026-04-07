"""Inference runner for inbox_env tasks."""

from __future__ import annotations

import json
import os
from uuid import uuid4

from openai import OpenAI

from client import InboxEnv
from models import InboxAction


SYSTEM_PROMPT = """
You are an email triage agent.
Always return strict JSON only with this schema:
{"action": "label", "email_id": "e3", "value": "urgent"}
Allowed actions: read,label,reply,route,delete,done
""".strip()


def _parse_action(raw: str) -> InboxAction:
    try:
        data = json.loads(raw)
        return InboxAction(
            action_type=data.get("action", "done"),
            email_id=data.get("email_id", ""),
            value=data.get("value"),
        )
    except Exception:
        return InboxAction(action_type="done", email_id="", value=None)


def _run_task(env: InboxEnv, client: OpenAI, model_name: str, task_id: str, seed: int) -> None:
    episode = str(uuid4())
    print(f"[START] task={task_id} episode={episode}")
    result = env.reset(seed=seed, task_id=task_id, episode_id=episode)
    done = False

    while not done:
        obs = result.observation
        user_msg = {
            "task_id": task_id,
            "step_count": obs.step_count,
            "budget_remaining": obs.budget_remaining,
            "inbox_summary": obs.inbox_summary,
            "action_result": obs.action_result,
            "email_body": obs.email_body,
        }
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_msg)},
            ],
            temperature=0.0,
        )
        raw = completion.choices[0].message.content or '{"action":"done","email_id":"","value":null}'
        action = _parse_action(raw)
        result = env.step(action)
        done = bool(result.done)
        reward = float(result.reward or 0.0)
        print(
            f"[STEP] step={result.observation.step_count} "
            f"action={action.action_type}:{action.email_id}:{action.value} "
            f"reward={reward:.2f} done={'true' if done else 'false'}"
        )

    score = float(result.reward or 0.0)
    print(f"[END] task={task_id} score={score:.2f} success={'true' if score >= 0.70 else 'false'}")


def main() -> None:
    api_base_url = os.environ["API_BASE_URL"]
    model_name = os.environ["MODEL_NAME"]
    hf_token = os.environ["HF_TOKEN"]
    llm = OpenAI(base_url=api_base_url, api_key=hf_token)

    with InboxEnv(base_url=api_base_url) as env:
        _run_task(env, llm, model_name, "1", seed=11)
        _run_task(env, llm, model_name, "2", seed=22)
        _run_task(env, llm, model_name, "3", seed=33)


if __name__ == "__main__":
    main()
