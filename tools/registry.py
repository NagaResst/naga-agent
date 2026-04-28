import importlib
import os


def discover_tools() -> tuple:
    tools_dir = os.path.dirname(__file__)
    skip = {"__init__.py", "registry.py"}

    tool_definitions = []
    tool_executors = {}

    for filename in os.listdir(tools_dir):
        if not filename.endswith(".py") or filename in skip:
            continue
        module_name = f"tools.{filename[:-3]}"
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"警告：加载工具模块 {module_name} 失败：{e}")
            continue

        definition = getattr(module, "TOOL_DEFINITION", None)
        executor = getattr(module, "execute", None)

        if definition is None or executor is None:
            continue

        tool_definitions.append(definition)
        tool_executors[definition["function"]["name"]] = executor

    return tool_definitions, tool_executors
