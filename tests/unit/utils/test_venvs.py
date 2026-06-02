# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nemo_rl.utils.venvs import (
    create_local_venv,
    get_nemo_rl_venv_dir,
    get_virtual_env_for_python_executable,
)
from tests.unit.conftest import TEST_ASSETS_DIR


def test_create_local_venv():
    # The temporary directory is created within the project.
    # For some reason, creating a virtual environment outside of the project
    # doesn't work reliably.
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        # Mock os.environ to set NEMO_RL_VENV_DIR for this test
        with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": tempdir}):
            venv_python = create_local_venv(
                py_executable="uv run --group docs", venv_name="test_venv"
            )
            assert os.path.exists(venv_python)
            assert venv_python == f"{tempdir}/test_venv/bin/python"
            # Check if sphinx package is installed in the created venv

            # Run a Python command to check if sphinx can be imported
            result = subprocess.run(
                [
                    venv_python,
                    "-c",
                    "import sphinx; print('Sphinx package is installed')",
                ],
                capture_output=True,
                text=True,
            )

            # Verify the command executed successfully (return code 0)
            assert result.returncode == 0, f"Failed to import sphinx: {result.stderr}"
            assert "Sphinx package is installed" in result.stdout


def test_get_nemo_rl_venv_dir_prefers_home_for_mnt_data_repo():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("nemo_rl.utils.venvs.git_root", "/mnt/data/sgsilva/nemo-rl-vlm"),
        patch("nemo_rl.utils.venvs.Path.home", return_value=Path("/home/sgsilva")),
    ):
        assert get_nemo_rl_venv_dir() == "/home/sgsilva/nemo-rl-vlm-ray-venvs"


def test_get_nemo_rl_venv_dir_respects_override():
    with patch.dict(
        os.environ, {"NEMO_RL_VENV_DIR": "/tmp/custom-ray-venvs"}, clear=True
    ):
        assert get_nemo_rl_venv_dir() == "/tmp/custom-ray-venvs"


def test_get_virtual_env_for_python_executable_detects_real_venv():
    with TemporaryDirectory() as tempdir:
        venv_root = Path(tempdir) / "demo_venv"
        bin_dir = venv_root / "bin"
        bin_dir.mkdir(parents=True)
        python_path = bin_dir / "python"
        python_path.write_text("")
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n")

        assert get_virtual_env_for_python_executable(str(python_path)) == str(
            venv_root
        )


def test_get_virtual_env_for_python_executable_ignores_non_venv_python():
    with TemporaryDirectory() as tempdir:
        python_path = Path(tempdir) / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")

        assert get_virtual_env_for_python_executable(str(python_path)) is None
