"""What the Converse client sends when a model rejects temperature, and what it reports, without AWS."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

from config import AWSConfig, LLMConfig
from core.llm_client import BedrockLLMClient


class ValidationException(Exception):
    pass


class FakeBedrock:
    """Records each Converse request; rejects temperature like GPT-5.6 Terra when asked to."""

    exceptions = SimpleNamespace(ValidationException=ValidationException)

    def __init__(self, rejects_temperature):
        self.rejects_temperature = rejects_temperature
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        if self.rejects_temperature and "temperature" in request["inferenceConfig"]:
            raise ValidationException("This model doesn't support the temperature field.")
        return {"stopReason": "end_turn", "output": {"message": {"content": [{"text": "reply"}]}}}


def make_client(rejects_temperature):
    bedrock = FakeBedrock(rejects_temperature)
    session = SimpleNamespace(client=lambda *args, **kwargs: bedrock)
    with mock.patch("core.llm_client.create_boto3_session", return_value=session):
        client = BedrockLLMClient(LLMConfig(model_id="us.openai.gpt-5.6-terra", temperature=0.0, top_p=0.9),
                                  AWSConfig())
    return client, bedrock


def generate(client):
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        reply = client.generate("Synthetic prompt")
    return reply, log.getvalue()


class RejectedSamplingTests(unittest.TestCase):
    def test_dropping_rejected_settings_is_logged(self):
        client, bedrock = make_client(rejects_temperature=True)
        reply, log = generate(client)

        self.assertEqual(reply, "reply")
        self.assertIn("temperature", bedrock.requests[0]["inferenceConfig"])
        self.assertNotIn("temperature", bedrock.requests[1]["inferenceConfig"])
        self.assertIn("temperature", log)
        self.assertIn("us.openai.gpt-5.6-terra", log)

    def test_sampling_sent_reports_what_was_sent(self):
        client, _ = make_client(rejects_temperature=True)
        self.assertEqual(client.sampling_sent, {"temperature": 0.0, "top_p": 0.9})
        generate(client)
        self.assertEqual(client.sampling_sent, {"temperature": None, "top_p": None})

    def test_accepted_settings_are_kept_without_a_log_line(self):
        client, bedrock = make_client(rejects_temperature=False)
        _, log = generate(client)

        self.assertEqual(len(bedrock.requests), 1)
        self.assertEqual(client.sampling_sent, {"temperature": 0.0, "top_p": 0.9})
        self.assertEqual(log, "")


if __name__ == "__main__":
    unittest.main()
