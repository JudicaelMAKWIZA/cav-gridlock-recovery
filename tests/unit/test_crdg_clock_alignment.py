"""Une date de graphe ne peut pas antidater une preuve propre."""

import pytest
from cav_recovery.crdg import LocalCRDG
from cav_recovery.communication.reports import ReportError
from cav_recovery.d01.observer import EgoObserver
from tests.support.d01_fixtures import packet


@pytest.mark.parametrize("at_s", [0.2, 0.6, float("nan"), True])
def test_incompatible_graph_clock_is_rejected(at_s):
    diagnostic = EgoObserver().update(packet(0.4))
    with pytest.raises(ReportError, match="do not align"):
        LocalCRDG("V0").update(diagnostic, None, at_s)
