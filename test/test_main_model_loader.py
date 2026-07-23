import types
import unittest
from unittest.mock import patch

import main


class MainModelLoaderTest(unittest.TestCase):
    def setUp(self):
        main.model_instances.clear()

    def tearDown(self):
        main.model_instances.clear()

    def test_get_model_instance_loads_once_and_reuses_cache(self):
        class DummyModel:
            instances = 0

            def __init__(self):
                type(self).instances += 1

        config = {
            "dummy": {
                "class_file": "dummy_module",
                "class_name": "DummyModel",
            }
        }
        module = types.SimpleNamespace(DummyModel=DummyModel)
        with (
            patch.object(main.ModelListUtil, "get_config", return_value=config),
            patch.object(main.importlib, "import_module", return_value=module) as importer,
        ):
            first = main.get_model_instance("dummy")
            second = main.get_model_instance("dummy")

        self.assertIs(first, second)
        self.assertEqual(DummyModel.instances, 1)
        importer.assert_called_once_with("models.dummy_module")

    def test_get_model_instance_rejects_unknown_name(self):
        with patch.object(main.ModelListUtil, "get_config", return_value={}):
            with self.assertRaisesRegex(ValueError, "model_list.yaml not found"):
                main.get_model_instance("missing")

        with patch.object(
            main.ModelListUtil,
            "get_config",
            return_value={"known": {"class_file": "known", "class_name": "Known"}},
        ):
            with self.assertRaisesRegex(ValueError, "model not found"):
                main.get_model_instance("missing")

    def test_get_model_instance_wraps_import_errors_without_caching(self):
        config = {
            "broken": {
                "class_file": "missing_module",
                "class_name": "MissingModel",
            }
        }
        with (
            patch.object(main.ModelListUtil, "get_config", return_value=config),
            patch.object(
                main.importlib,
                "import_module",
                side_effect=ModuleNotFoundError("not installed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "models.missing_module"):
                main.get_model_instance("broken")

        self.assertNotIn("broken", main.model_instances)


if __name__ == "__main__":
    unittest.main()
