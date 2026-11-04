import json
import unittest
from tests.v1.j.calibrate import classify, summarize, calibrate, OFFLINE_REPLY
from agent.main_agent import SESSION_BUSY_REPLY
from agent.budget import BUDGET_REPLY


def response(text, terminal="done"):
    return "event: delta\ndata: " + json.dumps({"text": text}) + "\n\nevent: " + terminal + "\ndata: {}\n\n"


class CapacityReportTests(unittest.TestCase):
    def test_done_is_not_automatically_success(self):
        for text, expected in ((OFFLINE_REPLY,"fixture_success"),(SESSION_BUSY_REPLY,"session_busy"),(BUDGET_REPLY,"budget_stopped"),("订单尚未接入","unexpected_reply")):
            with self.subTest(expected=expected):
                self.assertEqual(classify(200,response(text),OFFLINE_REPLY),expected)
        self.assertEqual(classify(200,response(OFFLINE_REPLY,"error"),OFFLINE_REPLY),"stream_failure")
        self.assertEqual(classify(200,"",OFFLINE_REPLY),"incomplete")
        self.assertEqual(classify(429,"",OFFLINE_REPLY),"rate_limited")
        self.assertEqual(classify(504,"",OFFLINE_REPLY),"timeout")
        self.assertEqual(classify(503,"",OFFLINE_REPLY),"http_failure")

    def test_percentile_and_throughput_denominators(self):
        samples=[{"elapsed_ms":n,"outcome":"fixture_success" if n<=8 else "budget_stopped"} for n in range(1,11)]
        result=summarize(samples,2)
        self.assertEqual((result["p50_ms"],result["p95_ms"]),(5,10))
        self.assertEqual(result["requests_per_second"],5)
        self.assertEqual(result["fixture_success_per_second"],4)
        self.assertEqual(result["outcomes"]["budget_stopped"],2)


class CapacitySmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_modes_run_without_network_and_measure_overlap(self):
        for mode in ("transport","graph"):
            with self.subTest(mode=mode):
                result=await calibrate(mode,2,1)
                self.assertEqual(result["outcomes"],{"fixture_success":2})
                self.assertEqual(result["actual_client_peak_in_flight"],2)
                self.assertEqual(result["actual_server_peak_streams"],2)
                if mode=="graph":
                    self.assertGreater(result["checkpoint_rows"],0)
                    self.assertGreater(result["sqlite_file_bytes_after"],0)
