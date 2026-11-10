import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from acceleration import describe_policy_config
from analysis.run_acceleration import main


class RecordingBackend:
    def __init__(self):
        self.calls = []
        self.closed = False

    def generate_for_policy(self, prompt, *, policy, plan, gen_length, seed, system_prompt=None):
        config = describe_policy_config(policy, plan=plan, gen_length=gen_length, seed=seed)
        self.calls.append((prompt, plan, policy, config['threshold']))
        return {'plan': plan, 'policy': policy, 'mode': config['mode'], 'config': config,
                'nfe': 1, 'residual_masks': 0, 'completion_status': 'complete'}

    def close(self):
        self.closed = True


class PlanEntrypointsTest(unittest.TestCase):
    def test_cli_reports_and_dispatches_all_six_effective_presets(self):
        with TemporaryDirectory() as directory:
            for plan in ('A', 'B'):
                for policy in ('FAST', 'NORMAL', 'SAFE'):
                    with self.subTest(plan=plan, policy=policy):
                        args = ['--plan', plan, '--policy', policy, '--gen-length', '64']
                        expected = describe_policy_config(policy, plan=plan, gen_length=64)
                        captured = io.StringIO()
                        with contextlib.redirect_stdout(captured):
                            self.assertEqual(main([*args, '--describe-config']), 0)
                        self.assertEqual(json.loads(captured.getvalue())['config'], expected)
                        backend = RecordingBackend()
                        path = Path(directory, f'{plan}-{policy}.jsonl')
                        with contextlib.redirect_stderr(io.StringIO()):
                            code = main([*args, '--prompt', 'A simple prompt', '--output', str(path)],
                                        backend_factory=lambda *a, **k: backend)
                        self.assertEqual(code, 0)
                        self.assertEqual(backend.calls, [('A simple prompt', plan, policy, expected['threshold'])])
                        row = json.loads(path.read_text(encoding='utf-8'))
                        self.assertEqual(row['plan'], plan)
                        self.assertEqual(row['generation']['config'], expected)
                        self.assertTrue(backend.closed)

    def test_plan_cannot_be_silently_ignored_by_direct_mode(self):
        for mode in ('fastdllm', 'compare', 'compare-all'):
            with self.subTest(mode=mode), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['--plan', 'B', '--mode', mode, '--describe-config']), 1)


if __name__ == '__main__':
    unittest.main()
