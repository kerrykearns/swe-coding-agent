"""The agent's "head": prompts, LLM access, and the reasoning strategies.

Kept import-light on purpose. ``python -m agent.core.baseline`` imports this
package before executing the module, so pulling the CLI's dependencies in here
would load them twice.
"""
