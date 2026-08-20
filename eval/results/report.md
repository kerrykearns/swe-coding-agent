# M8 Evaluation Report

## Summary

| Condition | Runs | Success rate | Avg turns | Avg tokens | Avg wall-clock (s) |
|---|---|---|---|---|---|
| baseline | 10 | 90.0% (9/10) | n/a (single-shot) | 1653 | 5.1 |
| react | 10 | 100.0% (10/10) | 7.8 | 30282 | 183.7 |

## Success rate by difficulty

| Condition | trivial | medium | hard |
|---|---|---|---|
| baseline | 100.0% (3/3) | 100.0% (3/3) | 75.0% (3/4) |
| react | 100.0% (3/3) | 100.0% (3/3) | 100.0% (4/4) |

## Methodology and limitations

Results based on 10 for baseline, 10 for react task run(s). A sample this small means individual results carry more weight than the percentages alone suggest — treat these as directional findings, not statistically robust claims.

Every `success` value here comes from independently re-running the task's own `verify_command` against the agent's final workspace state, never from the agent's own claim of success (see `eval/harness.py`'s "verified, not claimed" principle).


## Key finding

On a 10-task evaluation spanning three difficulty tiers, the single-shot
baseline and multi-turn ReAct agent were tied on trivial and
medium-difficulty bugs (100% success, both conditions). On hard-tier tasks
— each containing two independent defects in a single function — baseline
succeeded on 3 of 4 (75%), while the ReAct agent succeeded on all 4 (100%).
In every case, the ReAct agent's win came from catching a secondary defect
(e.g., missing input validation) that the baseline's one-shot fix missed
while addressing the primary bug correctly. This came at a substantial
cost: react used ~18.3x more tokens and ~36x more wall-clock time per task
on average. Overall, these results suggest multi-turn iteration's value
concentrates specifically on tasks with multiple independent failure
modes, at a cost that may not be justified for simpler bugs — a real
tradeoff any production deployment of an agentic coding system would need
to weigh.