import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, OPENAI_API_KEY="test-key", LANGSMITH_TRACING="false"):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.dict(data_service.PROSPECTS, deepcopy(data_service.PROSPECTS))
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.records.start()
        self.profiles.start()
        self.addCleanup(self.records.stop)
        self.addCleanup(self.profiles.stop)

    def test_update_persists_technology(self):
        result = data_service.update_prospect_info("LEAD-39002", "Terraform")

        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.fetch_tech_stack("LEAD-39002"))

    def test_profile_includes_update_with_and_without_cached_profile(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                data_service.PROSPECTS["LEAD-39002"]["tech_stack"] = ["AWS"]
                data_service._PROFILES.clear()
                if cached:
                    gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})

                data_service.update_prospect_info("LEAD-39002", "Terraform")
                result = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})

                self.assertIn("Terraform", result["prospect_profile"]["tech_stack"])

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        result = data_service.update_prospect_info("LEAD-39002", "Terraform")

        self.assertTrue(result["updated"])
        self.assertEqual(data_service.fetch_tech_stack("LEAD-39002").count("Terraform"), 1)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("unknown", "Terraform"),
            {"updated": False, "found": False},
        )
        self.assertIsNone(data_service.get_prospect_record("unknown"))

    def test_score_uses_saved_update_even_with_stale_profile(self):
        profile = gtm_agent.build_prospect_profile.invoke(
            {"prospect_id": "LEAD-39002"}
        )["prospect_profile"]
        self.assertNotIn("Terraform", profile["tech_stack"])
        gtm_agent.update_prospect_info.invoke(
            {"prospect_id": "LEAD-39002", "technology": "Terraform"}
        )

        def score(messages):
            scored_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
            self.assertIn("Terraform", scored_profile["tech_stack"])
            return gtm_agent.ProspectScore(
                score=100,
                justification="Terraform is present.",
                rubric_breakdown=gtm_agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100, segment_fit=100
                ),
            )

        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score
            result = gtm_agent.score_prospect.invoke({
                "prospect_profile": profile,
                "offering": data_service.get_offering("OFFER-10004"),
            })

        self.assertEqual(result["justification"], "Terraform is present.")


if __name__ == "__main__":
    unittest.main()
