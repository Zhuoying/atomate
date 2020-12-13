import re

import json

from datetime import datetime
from monty.json import MontyEncoder, jsanitize

from pathlib import Path

from monty.serialization import dumpfn, loadfn

import numpy as np
from amset.io import load_mesh
from amset.util import cast_dict_list
from atomate.utils.utils import get_meta_from_structure, env_chk, get_logger
from atomate.vasp.database import VaspCalcDb
from fireworks import FiretaskBase, explicit_serialize
from pymatgen.io.vasp import BSVasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = get_logger(__name__)


@explicit_serialize
class AmsetToDb(FiretaskBase):
    """
    Stores an AMSET run in the database.

    Assumes the current directory contains:

    - "settings.yaml" file containing input settings.
    - "transport_*.json" file containing transport properties.

    Optional params:
        db_file (str): Path to file containing the database credentials.
            Supports env_chk.
        additional_fields (dict): Fields added to the document such as user-defined
            tags or name, ids, etc.
    """

    optional_params = ["db_file", "additional_fields"]

    def run_task(self, fw_spec):
        db_file = env_chk(self.get('db_file'), fw_spec)
        doc = self.get("additional_fields") or {}
        converged = fw_spec.get("converged", None)

        settings = loadfn("settings.yaml")
        transport_file = next(Path(".").glob("*transport_*"))
        transport = loadfn(transport_file)
        inter_mesh = re.findall(r"transport_(\d+)x(\d+)x(\d+)\.", transport_file.name)
        inter_mesh = list(map(int, inter_mesh[0]))
        log = Path("amset.log").read_text()

        doc.update(
            {
                "calc_dir": str(Path.cwd()),
                "input": settings,
                "transport": transport,
                "created_at": datetime.utcnow(),
                "converged": converged,
                "kpoint_mesh": inter_mesh,
                "nkpoints": np.product(inter_mesh),
                "log": log,
            }
        )

        try:
            doc["timing"] = loadfn("timing.json.gz")
        except FileNotFoundError:
            pass

        # insert mesh if calculation is converged or convergence is not known
        mesh = None
        mesh_files = list(Path(".").glob("*mesh_*"))
        if len(mesh_files) > 0:
            mesh_data = load_mesh(mesh_files[0])
            doc.update(
                {
                   "is_metal": mesh_data["is_metal"],
                   "scattering_labels": mesh_data["scattering_labels"],
                   "soc": mesh_data["soc"],
                }
            )
            if converged is not False:
                # only store mesh if results not converged or converged not set
                mesh = _mesh_to_json(mesh_data)

        # insert structure information
        bs = _get_band_structure()
        structure = bs.structure
        sg = SpacegroupAnalyzer(structure, 0.01)
        doc.update(
            {
                "structure": structure.as_dict(),
                "formula_pretty": structure.composition.reduced_formula,
                "spacegroup": {
                    "symbol": sg.get_space_group_symbol(),
                    "number": sg.get_space_group_number(),
                    "point_group": sg.get_point_group_symbol(),
                    "source": "spglib",
                    "crystal_system": sg.get_crystal_system(),
                    "hall": sg.get_hall()
                },
            }
        )
        doc.update(get_meta_from_structure(structure))

        if db_file:
            mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

            if mesh:
                fsid, _ = mmdb.insert_gridfs(mesh, collection="amset_fs", compress=True)
                doc["amset_fs_id"] = fsid

            mmdb.db.amset.insert(jsanitize(doc, allow_bson=True))
        else:
            dumpfn(doc, "amset.json")


def _get_band_structure():
    """Find amset input file in current directory and extract band structure."""
    vr_files = list(Path(".").glob("*vasprun.xml*"))
    bs_files = list(Path(".").glob("*band_structure_data*"))

    if len(vr_files) > 0:
        return BSVasprun(str(vr_files[0])).get_band_structure()
    elif len(bs_files) > 0:
        return loadfn(bs_files[0])["band_structure"]

    raise ValueError("Could not find amset input in current directory.")


def _mesh_to_json(mesh):
    """Convert amset mesh data to json."""
    mesh = cast_dict_list(mesh)
    return json.dumps(mesh, cls=MontyEncoder)
