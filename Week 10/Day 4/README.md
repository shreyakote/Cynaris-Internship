# Human-in-the-Loop with LangGraph

## Objective

Build a stateful LangGraph agent with:

- Classify node
- Route node
- Respond node
- Conditional routing
- Human-in-the-loop interrupt
- Resume after human input

## Graph Flow

START
↓
Classify
↓
Route
↓
Respond
↓
END

The route is selected using the classification result.

## Classifications

- Technical
- General
- Support

## Human-in-the-Loop

Support requests trigger a LangGraph interrupt.

The graph pauses and waits for human input.

After the human provides a response, the graph resumes using the same thread ID.

## Example

Input:

I have a payment issue

The graph classifies the request as:

support

The graph pauses for human input.

After receiving:

Your payment issue has been received.

The graph resumes and returns the human response.

## Testing

Run:

```powershell
pytest -q