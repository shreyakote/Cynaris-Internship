# Week 10 Day 1 — LangGraph Stateful Agent Graphs

## Objective

Build a stateful LangGraph agent with classification, conditional routing, and human-in-the-loop interaction.

## Graph

START
  |
  v
Classify
  |
  v
Route
  |
  +---- Technical ----+
  |                   |
  +---- General ------+--> Respond --> END
  |
  +---- Support ------> Human Interrupt
                              |
                              v
                         Human Input
                              |
                              v
                            Resume
                              |
                              v
                           Respond

## Nodes

1. Classify
   - Identifies the request as technical, general, or support.

2. Route
   - Selects the route based on the classification.

3. Respond
   - Generates the response.
   - Support requests trigger a human-in-the-loop interrupt.

## Features

- LangGraph StateGraph
- Typed state
- Conditional edges
- InMemorySaver checkpointing
- Human-in-the-loop interrupt
- Resume using Command
- Automated pytest tests

## Testing

Run:

```powershell
pytest -q