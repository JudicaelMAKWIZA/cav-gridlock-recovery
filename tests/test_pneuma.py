from pathlib import Path
import tempfile
import unittest

from cav_recovery.empirical.pneuma import read_candidates


FIXTURES = Path(__file__).parent / "fixtures" / "pneuma"


class PneumaReaderTests(unittest.TestCase):
    def test_known_shape_and_line_numbers(self):
        rows = list(read_candidates(FIXTURES / "known.csv"))
        self.assertEqual([(row.line, row.groups, row.decomposable) for row in rows], [(2, 2, True), (3, 2, True)])

    def test_bom_spaces_and_terminal_separator_are_accepted(self):
        content = "\ufefftrack_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time\r\n x ; car ; 1 ; 36 ; 1 ; 2 ; 36 ; 0 ; 0 ; 0 ;\r\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            source.write_text(content, encoding="utf-8")
            row = next(read_candidates(source))
        self.assertTrue(row.decomposable)
        self.assertEqual(row.groups, 1)

    def test_final_empty_time_is_not_discarded_as_separator(self):
        content = "track_id; type; traveled_d; avg_speed; lat; lon; speed; lon_acc; lat_acc; time\nx;car;1;1;1;2;3;0;0;\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            source.write_text(content, encoding="utf-8")
            row = next(read_candidates(source))
        self.assertTrue(row.decomposable)
        self.assertEqual(row.fields[-1], "")
