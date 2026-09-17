from pathlib import Path

from research_agent import (
    crew,
    generate_report,
    researcher,
    reviewer,
    writer,
)
from tools import calculator, web_search


def test_researcher_agent_exists():
    assert researcher.role == "Researcher"


def test_writer_agent_exists():
    assert writer.role == "Writer"


def test_reviewer_agent_exists():
    assert reviewer.role == "Reviewer"


def test_crew_contains_three_agents():
    assert len(crew.agents) == 3


def test_crew_contains_three_tasks():
    assert len(crew.tasks) == 3


def test_calculator():
    assert calculator("10 + 5") == "15"


def test_calculator_multiplication():
    assert calculator("10 * 5") == "50"


def test_web_search():
    result = web_search("RAG")
    assert "RAG" in result


def test_generate_report_function_exists():
    assert callable(generate_report)


def test_report_file_path():
    path = Path("research_report.txt")
    assert path.name == "research_report.txt"