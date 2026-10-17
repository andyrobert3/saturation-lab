"""Checks for the knee detector, the outstanding-work series, and the metric parser."""

import unittest

from analysis.analyze import Req, detect_knee, outstanding_series, trend_per_second
from analysis.queue_theory import kingman_mean_wait, waits
from scripts.sweep import parse_metrics


class OpenSweepTests(unittest.TestCase):
    def test_outstanding_counts_sent_not_finished_and_excludes_clips(self):
        def req(actual, finished, status="ok"):
            return Req(0, 0, actual, 0, 0, finished, 16, status, False)

        requests = [req(100_000_000, 600_000_000),
                    req(400_000_000, 1_200_000_000),
                    req(0, 0, "clipped")]
        series = outstanding_series(requests, 2)
        self.assertEqual(series[0][1], 0)
        self.assertEqual(series[1][1], 1)
        self.assertEqual(series[4][1], 2)
        self.assertEqual(series[6][1], 1)
        self.assertEqual(series[12][1], 0)
        self.assertEqual(trend_per_second([(0, 0), (1, 1), (2, 2)], 0), 1.0)

    def test_server_metric_parser_sums_waiting_across_engines(self):
        body = ('vllm:num_requests_waiting{engine="0"} 2\n'
                'vllm:num_requests_waiting{engine="1"} 3\n'
                'vllm:num_requests_running{engine="0"} 4\n'
                'vllm:kv_cache_usage_perc{engine="0"} 0.4\n'
                'vllm:kv_cache_usage_perc{engine="1"} 0.7\n')
        values = parse_metrics(body)
        self.assertEqual(values["waiting"], 5)
        self.assertEqual(values["running"], 4)
        self.assertEqual(values["kv_cache_usage"], 0.7)

    def test_failed_or_erroring_rate_cannot_set_the_knee(self):
        def run(served, failed=False, errors=False):
            return {"row": {"served_rate_hz": served}, "failed": failed,
                    "clean": not failed and not errors}

        groups = {10.0: [run(10.0)], 12.0: [run(5.0, failed=True)],
                  13.0: [run(11.0)], 14.0: [run(5.0, errors=True)]}
        self.assertEqual(detect_knee(groups), (13.0, 11.0))
        self.assertEqual(detect_knee({10.0: [run(10.0)], 12.0: [run(5.0, errors=True)]}),
                         (None, 10.0))

    def test_queue_simulator_matches_exact_formulas(self):
        import numpy as np
        rng = np.random.default_rng(1)
        # M/D/1, Pollaczek-Khinchine: mean wait = rho * D / (2 * (1 - rho)) = 0.5 at rho 0.5.
        a = np.cumsum(rng.exponential(1 / 0.5, 400_000))
        self.assertAlmostEqual(waits(a, servers=1, service_s=1.0).mean(), 0.5, delta=0.02)
        # Uniform arrivals below capacity never wait; Kingman agrees (Ca² = Cs² = 0).
        self.assertEqual(waits(np.arange(1000) * 0.3, servers=4, service_s=1.0).max(), 0.0)
        self.assertEqual(kingman_mean_wait(3.0, 0.0, servers=4, service_s=1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
