"""Agent integration checks with a controlled model and no external services."""

import importlib
import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "LANGSMITH_TRACING": "false",
        })
        cls.environment.start()
        cls.addClassCleanup(cls.environment.stop)
        cls.agent = importlib.import_module("gtm_agent.gtm_agent")

    def test_application_tools_and_runtime_injection(self):
        rep = self.agent.data_service.get_rep("rep_amills")
        prospect = self.agent.data_service.get_prospect_record("LEAD-12853")
        calls = [
            ("get_current_rep", {}),
            ("lookup_offering", {"offering_id": "OFFER-10001"}),
            ("get_prospect", {"prospect_id": "LEAD-12853"}),
            ("send_prospect_email", {
                "prospect": {"name": prospect["name"], "email": prospect["email"]},
                "subject": "Demo next week",
                "body": "Please share your availability.",
            }),
        ]
        requests = []

        def generate(messages, **kwargs):
            requests.append(kwargs["tools"])
            index = len(requests) - 1
            if index < len(calls):
                name, args = calls[index]
                message = AIMessage(content="", tool_calls=[{
                    "name": name, "args": args, "id": str(index),
                }])
            else:
                message = AIMessage(content="Email sent.")
            return ChatResult(generations=[ChatGeneration(message=message)])

        with patch.object(ChatOpenAI, "_generate", side_effect=generate):
            result = self.agent.run_agent(
                "Look up OFFER-10001 and email LEAD-12853 about a demo next week.",
                user_id=rep["rep_id"],
            )

        outputs = {
            message.name: json.loads(message.content)
            for message in result["messages"] if isinstance(message, ToolMessage)
        }
        self.assertEqual(result["reply"], "Email sent.")
        self.assertEqual(outputs["get_current_rep"], {"rep": rep, "found": True})
        self.assertEqual(outputs["lookup_offering"]["offering"]["offering_id"], "OFFER-10001")
        self.assertEqual(outputs["get_prospect"]["prospect"]["email"], prospect["email"])
        email = outputs["send_prospect_email"]
        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["to"], prospect["email"])
        self.assertEqual(email["from"], rep["email"])
        self.assertEqual(email["from_name"], rep["name"])
        self.assertEqual(email["subject"], "Demo next week")
        self.assertEqual(email["body"], "Please share your availability.")
        expected_tools = {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        }
        for tools in requests:
            self.assertEqual({tool["function"]["name"] for tool in tools}, expected_tools)
            for tool in tools:
                self.assertNotIn("runtime", tool["function"]["parameters"].get("properties", {}))


if __name__ == "__main__":
    unittest.main()
