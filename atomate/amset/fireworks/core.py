from typing import Optional, Dict, Any, Union, List

from atomate.amset.firetasks.glue_tasks import CheckConvergence, CopyInputs, \
    ResubmitUnconverged
from atomate.amset.firetasks.parse_outputs import AmsetToDb
from atomate.amset.firetasks.run_calc import RunAmset
from atomate.amset.firetasks.write_inputs import UpdateSettings, WriteInputsFromMp
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.config import DB_FILE
from fireworks import Firework


class AmsetFW(Firework):
    def __init__(
        self,
        input_source: str,
        settings: Optional[Dict[str, Any]] = None,
        convergence_tol=0.1,
        resubmit: bool = False,
        db_file: str = DB_FILE,
        additional_fields: Optional[Dict[str, Any]] = None,
        parents: Optional[Union[Firework, List[Firework]]] = None,
        name: str = "amset",
        **kwargs
    ):
        """
        Firework to run amset.

        Args:
            input_source: The source for amset inputs (including settings, and vasprun/
                band structure/wavefunction files). Options are:

                - An mp_id, e.g., "mp_149", in which case the band structure from
                  Materials Project will be used as the input. Note, with this option,
                  `settings` must also be specified explicitly.
                - "workflow": Determine all amset inputs from calculations in the
                  current workflow. The vasprun (and wavefunction) will be obtained
                  from the most recent NSCF calculation, materials properties will be
                  determined from DFPT/elastic constant/deformation fireworks if they
                  are included. Note, this option requires `settings` to be specified
                  explicitly. Materials parameters not calculated as part of the
                  workflow can be specified in the settings dict.
                - "prev": Copy amset inputs from the previous calculation directory.
                  this can be selected when doing convergence of transport properties.
                  This option does not require settings to be specified explicitly.
            settings: AMSET settings, including materials parameters and temperature/
                doping selections. See the amset documentation for the available
                options.
            convergence_tol: Relative tolerance for checking amset property convergence.
            resubmit: Whether to submit an additional amset Firework with a larger
                interpolation factor after the current task.
            db_file: Path to file specifying db credentials.
            additional_fields: Fields added to the document such as
                user-defined tags or name, ids, etc.
            parents: Parents of this particular Firework. FW or list of FWS.
            name: Name for the Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """
        t = []
        additional_fields = additional_fields or {}
        if settings is None and input_source != "prev":
            raise ValueError(
                f"Settings must be specified for with {input_source} as input source"
            )

        if input_source == "workflow":
            raise NotImplemented("Workflow input source not yet implemented")
        elif input_source == "prev":
            t.append(CopyInputs(calc_loc=True))
        elif "mvs" in input_source or "mp" in input_source:
            t.append(WriteInputsFromMp(mp_id=input_source))
        else:
            raise ValueError(f"Unrecognised input source: {input_source}")

        if settings is not None:
            t.append(UpdateSettings(settings_updates=settings))

        t.append(UpdateSettings(settings_updates=">>amset_settings_updates<<"))
        t.append(RunAmset())

        if resubmit:
            t.append(CheckConvergence(tolerance=convergence_tol))

        t.append(PassCalcLocs(name="amset"))
        t.append(AmsetToDb(db_file=db_file, additional_fields=additional_fields))

        if resubmit:
            t.append(ResubmitUnconverged())

        super(AmsetFW, self).__init__(t, parents=parents, name=name, **kwargs)

