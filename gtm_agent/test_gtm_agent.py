import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")

from gtm_agent import data_service


class BillingPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.dict(data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def assert_no_billing(self, value, record):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for sensitive in record["billing_qualification"].values():
            self.assertNotIn(sensitive, serialized)

    def test_contact_allowlist(self):
        for pid, record in data_service.PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                with patch.dict(record, {"unexpected_field": "private"}):
                    result = agent.get_prospect.invoke({"prospect_id": pid})
                self.assertEqual(result, {
                    "found": True,
                    "prospect": {"prospect_id": pid, **{
                        key: record[key] for key in ("name", "email", "disqualified")
                    }},
                })

    def test_new_and_cached_profiles(self):
        for pid, record in data_service.PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                expected = {"prospect_id": pid, **{
                    k: v for k, v in record.items() if k != "billing_qualification"
                }}
                built = agent.build_prospect_profile.invoke({"prospect_id": pid})
                self.assertEqual(built, {"prospect_profile": expected, "found": True})
                self.assertEqual(data_service._PROFILES[pid], expected)
                data_service._PROFILES[pid] = {"prospect_id": pid, **record}
                cached = data_service.get_profile_from_db(pid)
                self.assertEqual(cached, {"prospect_profile": expected})
                self.assertEqual(
                    agent.build_prospect_profile.invoke({"prospect_id": pid}), built
                )
                self.assertIn("billing_qualification", record)

    def test_scoring_filters_supplied_profile(self):
        for pid, record in data_service.PROSPECTS.items():
            for include_id in (False, True):
                with self.subTest(prospect_id=pid, include_id=include_id):
                    profile = dict(record)
                    if include_id:
                        profile["prospect_id"] = pid
                        profile["tech_stack"] = ["unverified"]
                    offering = data_service.get_offering("OFFER-10005")
                    expected_score = {"score": 75, "justification": "Good fit"}
                    with patch.object(agent, "_scoring_llm") as scorer:
                        scorer.invoke.return_value.model_dump.return_value = expected_score
                        result = agent.score_prospect.invoke({
                            "prospect_profile": profile, "offering": offering,
                        })
                    self.assertEqual(result, expected_score)
                    messages = scorer.invoke.call_args.args[0]
                    self.assert_no_billing(messages, record)
                    prompt = messages[1]["content"]
                    supplied = json.loads(prompt.split("Prospect profile:\n", 1)[1])
                    self.assertEqual(supplied, {
                        **{k: v for k, v in profile.items() if k != "billing_qualification"},
                        "tech_stack": record["tech_stack"],
                    })
                    self.assertIn("billing_qualification", profile)

    def test_email_and_scoring_tool_workflows(self):
        runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        for pid, record in data_service.PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                rep = agent.get_current_rep.func(runtime)
                contact = agent.get_prospect.invoke({"prospect_id": pid})
                profile = agent.build_prospect_profile.invoke({"prospect_id": pid})
                offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10005"})
                arguments = {
                    "prospect_profile": profile["prospect_profile"],
                    "offering": offering["offering"],
                }
                self.assert_no_billing(arguments, record)
                with patch.object(agent, "_scoring_llm") as scorer:
                    scorer.invoke.return_value.model_dump.return_value = {"score": 75}
                    score = agent.score_prospect.invoke(arguments)
                email = agent.send_prospect_email.func(
                    contact["prospect"], "Demo", "Please schedule a demo.", runtime
                )
                self.assertEqual(email["status"], "sent")
                self.assertEqual(email["to"], record["email"])
                self.assertEqual(email["from"], rep["rep"]["email"])
                update = agent.update_prospect_info.invoke({
                    "prospect_id": pid, "technology": "Kafka",
                })
                self.assert_no_billing(
                    [rep, contact, profile, offering, score, email, update], record
                )

    def test_missing_records(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
        self.assertEqual(data_service.get_profile_from_db("missing"),
                         {"prospect_profile": None})


if __name__ == "__main__":
    unittest.main()
