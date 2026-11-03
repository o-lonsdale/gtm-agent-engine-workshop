"""Offline behavior tests for the GTM agent tools and request context."""

import json
import os
import unittest
from unittest.mock import Mock, patch

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as module


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.dict(module.data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.prospect_id = next(iter(module.data_service.PROSPECTS))
        self.offering_id = next(iter(module.data_service.OFFERINGS))
        self.prospect = module.data_service.get_prospect_record(self.prospect_id)
        self.rep = module.REP_IDS[0]
        self.score = module.ProspectScore(
            score=75,
            justification="Fit assessment",
            rubric_breakdown=module.RubricBreakdown(
                revenue_fit=80,
                tech_stack_match=70,
                segment_fit=75,
            ),
        )

    def run_tool_request(self, tool, args):
        model = ToolCallingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool.name,
                            "args": args,
                            "id": "request-tool",
                        }
                    ],
                ),
                AIMessage(content="Completed"),
            ]
        )
        graph = create_deep_agent(
            model=model,
            system_prompt=module.SYSTEM_PROMPT,
            tools=[
                module.lookup_offering,
                module.build_prospect_profile,
                module.get_prospect,
                module.get_current_rep,
                module.send_prospect_email,
                module.score_prospect,
            ],
        )
        with patch.object(module, "gtm_agent", wraps=graph) as agent:
            result = module.run_agent(
                "Complete this request", user_id=self.rep["rep_id"]
            )
        inputs = agent.invoke.call_args.args[0]
        context = inputs["messages"][0]
        self.assertEqual(context.type, "system")
        self.assertIn(self.rep["name"], context.content)
        self.assertIn(self.rep["email"], context.content)
        self.assertNotIn("always call the get_current_rep", module.SYSTEM_PROMPT)
        tool_messages = [
            message for message in result["messages"] if message.type == "tool"
        ]
        self.assertEqual([message.name for message in tool_messages], [tool.name])
        self.assertEqual(result["reply"], "Completed")
        return json.loads(tool_messages[0].content)

    def test_email_uses_signed_in_rep_without_lookup(self):
        result = self.run_tool_request(
            module.send_prospect_email,
            {
                "prospect": self.prospect,
                "subject": "Follow up",
                "body": "Hello!",
            },
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual(result["from_name"], self.rep["name"])
        self.assertEqual(result["to"], self.prospect["email"])
        self.assertEqual(result["body"], "Hello!")

    def test_score_loads_records_from_ids(self):
        with patch.object(module, "_scoring_llm") as scorer:
            scorer.invoke.return_value = self.score
            result = self.run_tool_request(
                module.score_prospect,
                {
                    "prospect_id": self.prospect_id,
                    "offering_id": self.offering_id,
                },
            )
        self.assertEqual(
            result,
            {
                **self.score.model_dump(),
                "prospect_name": self.prospect["name"],
                "offering_name": module.data_service.get_offering(self.offering_id)[
                    "name"
                ],
            },
        )
        payload = scorer.invoke.call_args.args[0][1]["content"]
        self.assertEqual(
            json.loads(payload.split("Prospect profile:\n", 1)[1])["tech_stack"],
            self.prospect["tech_stack"],
        )
        self.assertIn(self.prospect["name"], payload)
        self.assertEqual(
            module.data_service.get_profile_from_db(self.prospect_id)[
                "prospect_profile"
            ]["name"],
            self.prospect["name"],
        )

    def test_score_refreshes_cached_tech_stack(self):
        profile = module.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )
        module.data_service.save_profile_to_db(
            self.prospect_id,
            {
                **profile["prospect_profile"],
                "tech_stack": ["stale-technology"],
            },
        )
        with patch.object(module, "_scoring_llm") as scorer:
            scorer.invoke.return_value = self.score
            module.score_prospect.invoke(
                {"prospect_id": self.prospect_id, "offering_id": self.offering_id}
            )
        payload = scorer.invoke.call_args.args[0][1]["content"]
        self.assertNotIn("stale-technology", payload)
        self.assertEqual(
            json.loads(payload.split("Prospect profile:\n", 1)[1])["tech_stack"],
            self.prospect["tech_stack"],
        )

    def test_missing_prospect(self):
        with patch.object(module, "_scoring_llm") as scorer:
            result = module.score_prospect.invoke(
                {"prospect_id": "missing", "offering_id": self.offering_id}
            )
        self.assertEqual(result, {"prospect_profile": None, "found": False})
        scorer.invoke.assert_not_called()

    def test_invalid_offering(self):
        offering = module.data_service.get_offering(self.offering_id)
        invalid = [None] + [
            {**offering, field: value}
            for field, value in (
                ("required_tech_stack", []),
                ("min_annual_revenue", None),
                ("description", ""),
            )
        ]
        for record in invalid:
            with (
                self.subTest(record=record),
                patch.object(module.data_service, "get_offering", return_value=record),
                patch.object(module, "_scoring_llm") as scorer,
            ):
                result = module.score_prospect.invoke(
                    {"prospect_id": self.prospect_id, "offering_id": self.offering_id}
                )
                self.assertEqual(
                    result,
                    {"score": None, "error": "Cannot score without a valid offering."},
                )
                scorer.invoke.assert_not_called()

    def test_standalone_offering_lookup(self):
        self.assertEqual(
            self.run_tool_request(
                module.lookup_offering, {"offering_id": self.offering_id}
            ),
            {
                "offering": module.data_service.get_offering(self.offering_id),
                "found": True,
            },
        )
        self.assertEqual(
            module.lookup_offering.invoke({"offering_id": "missing"}),
            {"offering": None, "found": False},
        )

    def test_standalone_profile_build(self):
        result = self.run_tool_request(
            module.build_prospect_profile, {"prospect_id": self.prospect_id}
        )
        self.assertTrue(result["found"])
        self.assertEqual(
            result["prospect_profile"],
            {"prospect_id": self.prospect_id, **self.prospect},
        )
        self.assertEqual(
            module.build_prospect_profile.invoke({"prospect_id": self.prospect_id}),
            result,
        )
        self.assertEqual(
            module.build_prospect_profile.invoke({"prospect_id": "missing"}),
            {"prospect_profile": None, "found": False},
        )

    def test_unknown_rep_preserves_request(self):
        agent = Mock()
        agent.invoke.return_value = {"messages": [AIMessage(content="Done")]}
        with patch.object(module, "gtm_agent", agent):
            result = module.run_agent("Look up an offering", user_id="missing")
        self.assertEqual(
            agent.invoke.call_args.args[0]["messages"],
            [{"role": "user", "content": "Look up an offering"}],
        )
        self.assertEqual(result["reply"], "Done")


if __name__ == "__main__":
    unittest.main()
