import json

import zlib

import gridfs
from gridfs import NoFile
from pathlib import Path

from monty.serialization import dumpfn, loadfn

from pymatgen import Structure
from atomate.utils.utils import env_chk, get_logger
from atomate.vasp.database import VaspCalcDb
from fireworks import FiretaskBase, explicit_serialize
from pymatgen.electronic_structure.bandstructure import BandStructure

__author__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = get_logger(__name__)


@explicit_serialize
class WriteInputsFromMp(FiretaskBase):
    """
    Write band_structure_data.json input from MP band structure data.

    Will use the blessed NSCF uniform calculation if one is available.

    Required params:
        mp_id: The MP id of the material.
    """
    required_params = ["mp_id"]

    def run_task(self, fw_spec):
        mp_id = self["mp_id"]
        mp_db_file = env_chk(">>mp_db_file<<", fw_spec)
        calc_db = VaspCalcDb.from_db_file(mp_db_file, admin=False)

        # get uniform band structure task id from materials collection
        result = calc_db.db.materials.core.find_one(
            {"task_id": mp_id}, ["bandstructure.uniform_task"]
        )
        bs_id = result["bandstructure"]["uniform_task"]

        # get NELECT and structure from task
        result = calc_db.db.tasks.find_one(
            {"task_id": bs_id},
            ["calcs_reversed.output.outcar.nelect", "output.structure"]
        )
        nelect = result["calcs_reversed"][0]["output"]["outcar"]["nelect"]
        structure = Structure.from_dict(result["output"]["structure"])

        try:
            # get the band structure object with projections
            band_structure = calc_db.get_band_structure(task_id=bs_id)
        except gridfs.errors.NoFile:
            print(bs_id)
            logger.info(f"VaspCalcDb failed to get band structure for task {bs_id}. "
                        "Querying GridFS directly.")
            result = calc_db.db.bandstructure_fs.files.find_one(
                {"metadata.task_id": bs_id}, ["_id"]
            )
            fs = gridfs.GridFS(calc_db.db, "bandstructure_fs")
            bs_json = zlib.decompress(fs.get(result["_id"]).read())
            obj_dict = json.loads(bs_json.decode())
            band_structure = BandStructure.from_dict(obj_dict)

        # set the structure explicitly, as some band structure objects do not include it
        band_structure.structure = structure

        # dump the data
        data = {"band_structure": band_structure, "nelect": nelect}
        dumpfn(data, "band_structure_data.json")


@explicit_serialize
class WriteSettings(FiretaskBase):
    """
    Write amset settings file from settings dictionary.

    Required params:
        settings (dict): A dictionary of settings.
    """

    required_params = ["settings"]

    def run_task(self, fw_spec):
        dumpfn(self["settings"], "settings.yaml")


@explicit_serialize
class UpdateSettings(FiretaskBase):
    """
    Update amset settings from a dictionary.

    Required params:
        settings_updates (Union[str, dict]): A dictionary of settings updates or env
            chk string.
    """

    required_params = ["settings_updates"]

    def run_task(self, fw_spec):
        if Path("settings.yaml").exists():
            settings = loadfn("settings.yaml")
        else:
            settings = {}

        settings_updates = env_chk(self.get("settings_updates"), fw_spec)
        settings.update(settings_updates)
        dumpfn(settings, "settings.yaml")


@explicit_serialize
class UpdateSettingsEnvChk(FiretaskBase):
    """
    Update amset settings from a dictionary.

    Required params:
        settings_updates (dict): A dictionary of settings updates.
    """

    required_params = ["settings_updates"]

    def run_task(self, fw_spec):
        if Path("settings.yaml").exists():
            settings = loadfn("settings.yaml")
        else:
            settings = {}
        settings.update(self["settings_updates"])
        dumpfn(settings, "settings.yaml")
