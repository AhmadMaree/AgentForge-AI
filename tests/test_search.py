from unittest.mock import Mock

from agentforge.tools.search import SearchTool


def test_search_tool_class_metadata_and_execution():
    client = Mock()
    client.search.return_value = {
        "results": [
            {
                "title": "Result",
                "url": "https://example.com",
                "content": "A matching result",
                "score": 0.9,
            }
        ]
    }
    tool = SearchTool(client=client)

    result = tool.run({"query": "test"})

    assert SearchTool.name == "web_search"
    assert result == {
        "query": "test",
        "results": [
            {
                "title": "Result",
                "url": "https://example.com",
                "content": "A matching result",
                "score": 0.9,
            }
        ],
        "count": 1,
    }
    client.search.assert_called_once_with(
        query="test",
        search_depth="basic",
        max_results=5,
        include_answer=False,
    )
