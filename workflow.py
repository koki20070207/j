"""上限付きの低リスク複数ステップワークフロー実行。"""

from typing import Any, Dict, List

from operation_dispatcher import OperationDispatchError, dispatch_operation

MAX_WORKFLOW_STEPS = 5


def run_workflow(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not steps or len(steps) > MAX_WORKFLOW_STEPS:
        raise OperationDispatchError("ワークフローは1〜5ステップで指定してください。")
    results = []
    for step in steps:
        if not isinstance(step, dict) or "operation" not in step:
            raise OperationDispatchError("各ステップにはoperationが必要です。")
        result = dispatch_operation(step["operation"], step.get("request", {}))
        results.append(result)
    return {"status": "succeeded", "steps": len(results), "results": results}
