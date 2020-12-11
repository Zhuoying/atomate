import os

from pathlib import Path

import numpy as np
from monty.serialization import loadfn

from amset.util import tensor_average
from atomate.common.firetasks.glue_tasks import CopyFiles, get_calc_loc
from atomate.utils.utils import get_logger, env_chk
from fireworks import explicit_serialize, FiretaskBase, FWAction

_CONVERGENCE_PROPERTIES = ("mobility", "seebeck")
_INPUTS = [
    "settings.yaml",
    "vasprun.xml",
    "band_structure_data.json",
    "wavefunction.h5",
    "deformation.h5",
]
logger = get_logger(__name__)


@explicit_serialize
class CopyInputs(CopyFiles):
    """
    Copy amset input files from to the current directory.

    Note that you must specify either "calc_loc" or "calc_dir" to indicate
    the directory containing the input files.

    Optional params:
        calc_loc (Union[str, bool]): If True will set most recent calc_loc. If str
            search for the most recent calc_loc with the matching name
        calc_dir (str): Path to dir that contains amset output files.
        filesystem (str): Remote filesystem. e.g. username@host.
    """

    optional_params = ["calc_loc", "calc_dir", "filesystem"]

    def run_task(self, fw_spec):
        calc_loc = self.get("calc_loc") or {}
        calc_dir = self.get("calc_dir") or None
        filesystem = self.get("filesystem") or None

        if calc_loc:
            calc_loc = get_calc_loc(self["calc_loc"], fw_spec["calc_locs"])

        self.setup_copy(
            calc_dir,
            filesystem=filesystem,
            files_to_copy=_INPUTS,
            from_path_dict=calc_loc
        )
        self.copy_files()

    def copy_files(self):
        all_files = self.fileclient.listdir(self.from_dir)

        from_dir = Path(self.from_dir)
        to_dir = Path(self.to_dir)

        for f in self.files_to_copy:
            from_file = str(from_dir / f)
            to_file = str(to_dir / f)

            # handle gzipped files
            for ext in ["", ".gz", ".GZ"]:
                if f + ext in all_files:
                    self.fileclient.copy(from_file + ext, to_file + ext)


@explicit_serialize
class CheckConvergence(FiretaskBase):
    """
    Checks the convergence of amset transport properties.

    Expects calc_locs to be in the firework spec and that a transport file is in the
    current directory.

    Optional params:
        tolerance (float): Relative convergence tolerance. Default is `0.05` (i.e. 5 %).
        properties (list[str]): List of properties for which convergence is assessed.
            The calculation is only flagged as converged if all properties pass the
            convergence checks. Options are: "conductivity", "seebeck", "mobility",
            "electronic thermal conductivity". Default is `["mobility", "seebeck"]`.
    """

    optional_params = ["tolerance", "properties"]

    def run_task(self, fw_spec):
        tol = self.get("tolerance") or 0.05
        properties = self.get("properties") or _CONVERGENCE_PROPERTIES

        calc_locs = fw_spec.get("calc_locs", [])
        old_transport = None
        if len(calc_locs) > 0 and "amset" in calc_locs[-1]["name"]:
            transport_files = list(Path(calc_locs[-1]["path"]).glob("*transport_*"))

            if len(transport_files) > 1:
                old_transport = loadfn(transport_files[-1])

        if old_transport:
            # old calculation was found, we can now check for convergence
            new_transport = loadfn(next(Path(".").glob("*transport_*")))
            converged = _is_converged(new_transport, old_transport, tol, properties)
        else:
            converged = False

        return FWAction(update_spec={"converged": converged})


@explicit_serialize
class ResubmitUnconverged(FiretaskBase):
    """
    Detours to an amset calculation with a larger interpolation factor if unconverged.

    Expect the "converged" key to be in the firework spec.

    Optional params:
        interpolation_increase (int): Absolute amount by which to increase interpolation
            factor if resubmitting. Default is `10`.
    """

    optional_params = ["interpolation_increase"]

    def run_task(self, fw_spec):
        inter_inc = self.get("interpolation_increase") or 10
        converged = fw_spec.get("converged", True)

        if not converged:
            from atomate.amset.fireworks.core import AmsetFW

            settings = loadfn("settings.yaml")
            settings["interpolation_factor"] += inter_inc
            logger.info(
                "Resubmitting with interpolation_factor: "
                f"{settings['interpolation_factor']}"
            )
            fw = AmsetFW("prev", settings=settings, resubmit=True)

            # ensure to copy over fworker options to child firework
            # also, manually update calc locs
            # TODO: Also copy db_file, additional_fields etc
            fk = ["_fworker", "_category", "_queueadaptor", "calc_locs"]

            fw.spec.update({k: fw_spec[k] for k in fk if k in fw_spec})
            return FWAction(detours=[fw])
        else:
            logger.info("amset calculation is converged.")


def _is_converged(new_transport, old_transport, tol, properties):
    """Check if all transport properties (averaged) are converged within the tol."""
    converged = True
    for prop in properties:
        if prop not in new_transport and prop not in old_transport:
            logger.warning(f"'{prop}' not in new or old transport data, skipping...")
            continue

        new_prop = tensor_average(new_transport[prop])
        old_prop = tensor_average(old_transport[prop])
        diff: np.ndarray = (new_prop - old_prop) / new_prop
        diff[np.isnan(diff)] = 0

        if not np.all(diff <= tol):
            logger.info(f"{prop} is not converged: max diff: {np.max(diff) * 100} %")
            converged = False
    return converged



