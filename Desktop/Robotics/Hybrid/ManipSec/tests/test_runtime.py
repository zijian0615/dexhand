from __future__ import annotations

import unittest

from manipsec_bench import PolicyEnvView, SecurityEvent


class FakeRobot:
    action_dim = 8


class FakeEnv:
    device = "cpu"
    num_envs = 1
    dt = 1 / 60
    scene = object()
    robot = FakeRobot()
    iscene = object()

    def __init__(self) -> None:
        self.actions = []

    def get_states(self, env_ids=None):
        return {"scene": {}, "robot": {}}

    def describe(self):
        return "fixed pick and place"

    def describe_stage(self, root=None, raw=False):
        return []

    def step(self, action, render=False):
        self.actions.append(action)

    def reset(self, env_ids=None, *, seed=None):
        pass

    def set_states(self, states, env_ids=None):
        pass


class Validator:
    def validate(self, action, env, step):
        return (
            SecurityEvent(
                code="invalid_action",
                step=step,
                env_index=0,
                source="action_validator",
                attempted=True,
                realized=False,
                blocked=True,
            ),
        )


class EventSink:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class RuntimeTests(unittest.TestCase):
    def test_policy_view_exposes_step_but_not_state_writes(self) -> None:
        raw = FakeEnv()
        view = PolicyEnvView(raw)

        view.step([[0.0] * 8])

        self.assertEqual(view.steps, 1)
        self.assertEqual(len(raw.actions), 1)
        self.assertFalse(hasattr(view, "reset"))
        self.assertFalse(hasattr(view, "set_states"))

    def test_validation_events_reach_hidden_security_sink(self) -> None:
        sink = EventSink()
        view = PolicyEnvView(FakeEnv(), validator=Validator(), event_sinks=(sink,))

        view.step([[0.0] * 8])

        self.assertEqual(len(view.events), 1)
        self.assertEqual(sink.events, list(view.events))


if __name__ == "__main__":
    unittest.main()
