"""Basic tests for the AG2 plugin package."""


def test_register_module_imports():
    """Verify the entry point module loads without error."""
    from nat.plugins.ag2 import register  # noqa: F401


def test_framework_enum_has_ag2():
    """Verify AG2 is registered in the framework enum."""
    from nat.builder.framework_enum import LLMFrameworkEnum
    assert hasattr(LLMFrameworkEnum, "AG2")
    assert LLMFrameworkEnum.AG2 == "ag2"
