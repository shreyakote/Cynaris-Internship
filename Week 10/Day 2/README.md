# Week 10 Day 2 — LangGraph State Machines & Conditional Edges

## Objective

Build a stateful LangGraph agent with classification, conditional routing, and human-in-the-loop interaction.

## Graph Flow

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
  +---- General ------+----> Respond ----> END
  |
  +---- Support -----> Human Interrupt
                              |
                              v
                         Human Input
                              |
                              v
                            Resume
                              |
                              v
                           Respond
                              |
                              v
                             END

## Nodes

### 1. Classify

Classifies user requests into:

- technical
- support
- general

### 2. Route

Uses conditional edges based on the classification.

### 3. Respond

Returns a response for the selected route.

Support requests pause the graph using a human-in-the-loop interrupt.

## Features

- LangGraph StateGraph
- Typed state
- Conditional edges
- InMemorySaver checkpointing
- Human-in-the-loop interrupt
- Command-based resume
- Automated pytest tests

## Test Inputs

The graph is tested with five inputs:

1. Python error
2. LangGraph state
3. Capital of France
4. General question
5. API question

## Human-in-the-Loop

Support requests pause execution and wait for human input.

The graph is then resumed using:

```python
Command(resume="human response")