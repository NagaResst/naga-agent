import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent import context_window


class ContextWindowTests(unittest.TestCase):
    def test_metadata_is_cached_per_provider_and_model(self):
        client = Mock()
        client.models.retrieve.return_value = {"id": "model", "context_window": 32768}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(context_window, "_CACHE_FILE", os.path.join(directory, "windows.json")), \
             patch.object(context_window, "_CACHE_LOADED", False), \
             patch.object(context_window, "_DETECTED_WINDOWS", {}), \
             patch.object(context_window, "_DETECTED_AT", {}), \
             patch.object(context_window, "_PROBED_WINDOWS", {}):
            self.assertEqual(context_window.get_context_window("model", client=client, base_url="https://a/v1"), 32768)
            self.assertEqual(context_window.get_context_window("model", client=client, base_url="https://a/v1"), 32768)
            self.assertEqual(context_window.get_context_window("model", 16384, client=client, base_url="https://a/v1"), 16384)
            client.models.retrieve.assert_called_once_with("model")

            context_window._DETECTED_WINDOWS.clear()
            context_window._CACHE_LOADED = False
            self.assertEqual(context_window.get_context_window("model", base_url="https://a/v1"), 32768)
            self.assertEqual(context_window.get_context_window("model", client=client, base_url="https://b/v1"), 32768)
            self.assertEqual(client.models.retrieve.call_count, 2)

    def test_unavailable_metadata_retries_after_negative_cache_expires(self):
        client = Mock()
        client.models.retrieve.side_effect = RuntimeError("not supported")
        client.models.list.side_effect = RuntimeError("not supported")
        with patch.object(context_window, "_CACHE_LOADED", True), \
             patch.object(context_window, "_DETECTED_WINDOWS", {}), \
             patch.object(context_window, "_PROBED_WINDOWS", {}), \
             patch.object(context_window.time, "monotonic", side_effect=[100, 100, 120, 161, 161]):
            self.assertEqual(context_window.get_context_window("unknown", client=client), 131072)
            self.assertEqual(context_window.get_context_window("unknown", client=client), 131072)
            self.assertEqual(context_window.get_context_window("unknown", client=client), 131072)
        self.assertEqual(client.models.retrieve.call_count, 2)
        self.assertEqual(client.models.list.call_count, 2)

    def test_expired_positive_metadata_can_shrink_and_falls_back_on_probe_failure(self):
        client = Mock()
        client.models.retrieve.return_value = {"context_window": 32768}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(context_window, "_CACHE_FILE", os.path.join(directory, "windows.json")), \
             patch.object(context_window, "_CACHE_LOADED", False), \
             patch.object(context_window, "_DETECTED_WINDOWS", {}), \
             patch.object(context_window, "_DETECTED_AT", {}), \
             patch.object(context_window, "_PROBED_WINDOWS", {}):
            self.assertEqual(context_window.get_context_window("model", client=client), 32768)
            key = context_window._cache_key("model", None)
            context_window._DETECTED_AT[key] -= context_window._POSITIVE_CACHE_TTL + 1
            client.models.retrieve.return_value = {"context_window": 8192}
            self.assertEqual(context_window.get_context_window("model", client=client), 8192)
            context_window._DETECTED_AT[key] -= context_window._POSITIVE_CACHE_TTL + 1
            client.models.retrieve.side_effect = RuntimeError("metadata unavailable")
            client.models.list.side_effect = RuntimeError("metadata unavailable")
            self.assertEqual(context_window.get_context_window("model", client=client), 8192)


if __name__ == "__main__":
    unittest.main()