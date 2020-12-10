from monty.serialization import dumpfn
from fireworks import FiretaskBase, explicit_serialize
from amset.core.run import Runner

__author__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


@explicit_serialize
class RunAmset(FiretaskBase):
    """
    Run amset in the current directory.
    """

    def run_task(self, fw_spec):
        runner = Runner.from_directory(directory='.')
        _, usage_stats = runner.run(return_usage_stats=True)
        dumpfn(usage_stats, "timing.json.gz")
