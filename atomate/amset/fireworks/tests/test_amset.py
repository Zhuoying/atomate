from pathlib import Path

from fireworks import FWorker, Workflow
from fireworks.core.rocket_launcher import rapidfire
from atomate.utils.testing import AtomateTest
from atomate.amset.fireworks.core import AmsetFW

module_dir = Path(__file__).resolve().parent
db_dir = module_dir / "../../../common/test_files"
ref_dir = module_dir / "../../test_files"
mp_db_file = "/Users/alex/Google Drive/dev/notebooks/2020-12-AMSET-MP/mp_db.json"


class TestAmsetWF(AtomateTest):
    def setUp(self, lpad=True):
        super(TestAmsetWF, self).setUp(lpad=lpad)

        settings = {
            # general settings
            "doping": [-1e14, -1e15],
            "temperatures": [200, 300],
            "scissor": 2,
            "interpolation_factor": 2,
            "deformation_potential": (1.2, 8.6),
            "elastic_constant": 139.7,
            "donor_charge": 1,
            "acceptor_charge": 1,
            "static_dielectric": 12.18,
            "high_frequency_dielectric": 10.32,
            "pop_frequency": 8.16,
            "write_mesh": True,
            "use_projections": True,
            "nworkers": 2
        }
        amset_fw = AmsetFW("mp-2534", settings=settings, resubmit=True)
        self.wf = Workflow([amset_fw])

    def test_wf(self):

        fw_ids = self.lp.add_wf(self.wf)
        fworker = FWorker(env={"db_file": db_dir / "db.json", "mp_db_file": mp_db_file})

        if not Path(mp_db_file).exists():
            # skip tests if file not present
            return

        rapidfire(self.lp, fworker=fworker)

        # check workflow finished without error
        fw_id = list(fw_ids.values())[0]
        wf = self.lp.get_wf_by_fw_id(fw_id)
        is_completed = [s == "COMPLETED" for s in wf.fw_states.values()]
        self.assertTrue(all(is_completed))

