"""Check collection numbering and lossless episode extraction."""
import csv
import json
from pathlib import Path
import unittest
from add_episode import HERE, sha


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.document=json.loads((HERE/'mission_list.json').read_text())
        self.selected=[s for s in self.document['slots'] if s['status']=='selected']

    def test_numbering_and_categories(self):
        self.assertEqual([s['episode'] for s in self.document['slots']],list(range(1,len(self.document['slots'])+1)))
        self.assertEqual([s['source_episode'] for s in self.selected[:3]],[64,56,55])
        self.assertEqual([s['source_episode_18m'] for s in self.selected[:3]],[24,16,15])
        self.assertTrue(all(s['case']['distance_group_m']==18 for s in self.selected))
        self.assertEqual(len(rows(HERE/'mission_list.csv')),len(self.document['slots']))
        self.assertEqual(len(rows(HERE/'episodes.csv')),len(self.selected))

    def test_archived_measurements_match_originals(self):
        sources={s['source_run'] for s in self.selected}
        for source_name in sources:
            source=Path(source_name)
            selected=[s for s in self.selected if s['source_run']==source_name]
            wanted={s['source_episode'] for s in selected}
            for name in ('episodes.csv','trajectories.csv','power_samples.csv','fall_diagnostics.csv'):
                expected={episode:[] for episode in wanted}
                with (source/name).open(newline='') as stream:
                    for row in csv.DictReader(stream):
                        episode=int(row['episode'])
                        if episode in expected:
                            expected[episode].append(row)
                for slot in selected:
                    original=expected[slot['source_episode']]
                    actual=rows(HERE/slot['episode_directory']/name)
                    self.assertEqual(len(actual),len(original))
                    for a,b in zip(actual,original):
                        self.assertEqual(int(a['episode']),slot['episode'])
                        self.assertEqual(int(a['source_episode']),slot['source_episode'])
                        self.assertEqual({k:(str(slot['source_episode']) if k=='episode' else a[k]) for k in b},b)

    def test_review_summary_matches_registry(self):
        summary=json.loads((HERE/'category_summary.json').read_text())
        self.assertEqual(summary['total_saved'],len(self.selected))
        self.assertEqual(summary['total_tentative'],sum(s.get('review_status')=='tentative' for s in self.selected))
        self.assertEqual(summary['remaining_slots'],sum(s['status']=='pending' for s in self.document['slots']))
        self.assertEqual(summary['total_confirmed']+summary['total_tentative'],len(self.selected))
        for row in summary['categories']:
            selected=[s for s in self.selected if s['case']['category']==row['category']]
            self.assertEqual(row['saved'],len(selected))
            self.assertEqual(row['confirmed']+row['tentative'],row['saved'])

    def test_source_configuration_preserved(self):
        checked_sources=set()
        for slot in self.selected:
            folder=HERE/slot['episode_directory']
            provenance=json.loads((folder/'provenance.json').read_text())
            source=Path(slot['source_run'])
            self.assertEqual(provenance['source_config_sha256'],sha(source/'evaluation_config.json'))
            archived=HERE/provenance['archived_inputs']
            if str(archived) in checked_sources:
                continue
            checked_sources.add(str(archived))
            for path in (source/'inputs').rglob('*'):
                if path.is_file():
                    self.assertEqual(sha(path),sha(archived/path.relative_to(source/'inputs')))


if __name__=='__main__':
    unittest.main()
