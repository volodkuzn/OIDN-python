from __future__ import annotations

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BDistWheel(_bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _python, _abi, platform_tag = super().get_tag()
        return ("py3", "none", platform_tag)
