# Week 10 Day 3 — LangGraph + Memory

## Objective

Build a LangGraph state machine with persistent conversation memory.

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
  +---- Support ------+----> Respond ----> END
  |
  +---- General ------+

## Nodes

### 1. Classify

Classifies each user message as:

- technical
- support
- general

### 2. Route

Uses conditional edges to select the correct route.

### 3. Respond

Generates a response and stores the conversation history.

## Persistent Memory

The graph uses LangGraph `InMemorySaver` checkpointing.

A `thread_id` identifies a conversation.

Messages belonging to the same thread are stored together.

Example:

```python
chat("Hello", "conversation-1")
chat("How does Python work?", "conversation-1")