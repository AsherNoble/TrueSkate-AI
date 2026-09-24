"""Guard the exact multi-pointer payload used by the bounded XR2 probe."""

from scripts.inspect.probe_simultaneous_taps import send_pair
from scripts.inspect.probe_four_simultaneous_taps import FOUR_POINTS, send_four
from scripts.inspect.probe_grid_simultaneous_taps import GRID_POINTS, send_grid
from scripts.inspect.probe_eight_simultaneous_taps import EIGHT_POINTS, ORDER_NAMES, ordered_points, send_eight
from scripts.inspect.probe_five_simultaneous_taps import FIVE_POINTS, send_five


class FakeDriver:
    def __init__(self):
        self.commands = []

    def release_actions(self):
        pass

    def execute(self, command, payload):
        self.commands.append((command, payload))


def test_send_pair_sends_both_touch_sources_on_same_tick():
    driver = FakeDriver()

    send_pair(driver, width=414, height=896)

    assert len(driver.commands) == 1
    command, payload = driver.commands[0]
    assert command == "actions"
    sources = payload["actions"]
    assert len(sources) == 2
    assert len({source["id"] for source in sources}) == 2
    assert [(source["actions"][0]["x"], source["actions"][0]["y"])
            for source in sources] == [(190, 448), (223, 448)]
    assert all([action["type"] for action in source["actions"]]
               == ["pointerMove", "pointerDown", "pause", "pointerUp"]
               for source in sources)


def test_send_four_sends_square_in_one_command():
    driver = FakeDriver()

    send_four(driver)

    assert len(driver.commands) == 1
    command, payload = driver.commands[0]
    assert command == "actions"
    sources = payload["actions"]
    assert len(sources) == 4
    assert len({source["id"] for source in sources}) == 4
    assert [(source["actions"][0]["x"], source["actions"][0]["y"])
            for source in sources] == list(FOUR_POINTS)
    assert all([action["type"] for action in source["actions"]]
               == ["pointerMove", "pointerDown", "pause", "pointerUp"]
               for source in sources)


def test_send_grid_sends_91_safe_touches_on_same_tick():
    driver = FakeDriver()

    send_grid(driver)

    assert len(driver.commands) == 1
    command, payload = driver.commands[0]
    assert command == "actions"
    sources = payload["actions"]
    assert len(sources) == len(GRID_POINTS) == 91
    assert len({source["id"] for source in sources}) == 91
    assert [(source["actions"][0]["x"], source["actions"][0]["y"])
            for source in sources] == list(GRID_POINTS)
    assert all([action["type"] for action in source["actions"]]
               == ["pointerMove", "pointerDown", "pause", "pointerUp"]
               for source in sources)


def test_send_eight_keeps_all_points_under_each_ordering():
    for name in ORDER_NAMES:
        driver = FakeDriver()

        send_eight(driver, order=name)

        assert len(driver.commands) == 1
        command, payload = driver.commands[0]
        assert command == "actions"
        sources = payload["actions"]
        assert len(sources) == len(EIGHT_POINTS) == 8
        assert len({source["id"] for source in sources}) == 8
        points = [(source["actions"][0]["x"], source["actions"][0]["y"])
                  for source in sources]
        assert points == list(ordered_points(name))
        assert set(points) == set(EIGHT_POINTS)
        assert all([action["type"] for action in source["actions"]]
                   == ["pointerMove", "pointerDown", "pause", "pointerUp"]
                   for source in sources)


def test_send_five_uses_die_pattern_in_one_command():
    driver = FakeDriver()

    send_five(driver)

    assert len(driver.commands) == 1
    command, payload = driver.commands[0]
    assert command == "actions"
    sources = payload["actions"]
    assert len(sources) == 5
    assert len({source["id"] for source in sources}) == 5
    assert [(source["actions"][0]["x"], source["actions"][0]["y"])
            for source in sources] == list(FIVE_POINTS)
    assert all([action["type"] for action in source["actions"]]
               == ["pointerMove", "pointerDown", "pause", "pointerUp"]
               for source in sources)
