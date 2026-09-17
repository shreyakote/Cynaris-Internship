# Stateful Customer Support Agent

## Objective

Build a stateful customer support agent using LangGraph.

## Features

- Customer request classification
- Conditional routing
- Account support
- Technical support
- Payment support
- General support
- Conversation memory
- Human-in-the-loop review
- Interrupt and resume workflow
- Thread-based conversations

## Architecture

Customer Message
        |
        v
    Classify
        |
        v
      Route
     /     \
    /       \
Respond   Human Review
    |         |
    v      Interrupt
   END        |
              v
        Human Response
              |
              v
            Resume

## Categories

The agent identifies four categories:

1. Payment
2. Account
3. Technical
4. General

Payment-related requests are sent to human review.

## Stateful Conversations

LangGraph uses a thread ID to maintain conversation state.

Example:

```text
customer-1
    |
    +-- Customer message 1
    +-- Agent response 1
    +-- Customer message 2
    +-- Agent response 2