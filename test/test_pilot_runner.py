"""Check paid-run queue exclusions and failure isolation without running models/APIs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_runner', ROOT / 'evaluation/pilot/run_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotQueueTest(unittest.TestCase):
    def test_length_feedback_has_correct_direction(self):
        self.assertIn('Remove approximately 75 words',pilot.length_revision(775))
        self.assertIn('Add approximately 83 words',pilot.length_revision(617))

    def test_bertscore_nested_return_is_saved(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            run=Path(directory)
            pilot.write_json(run/'manifest.json',{'skipped_movies':['yes_man','get_out','the_martian']})
            out=run/'baby_driver'
            out.mkdir()
            for name in ['generated_summary.txt','reference_summary.txt']:
                (out/name).write_text('sample text',encoding='utf-8')
            pilot.write_json(out/'metrics.json',{})
            scalar=SimpleNamespace(item=lambda:0.8)
            score=Mock(return_value=(([scalar],[scalar],[scalar]),'test-hash'))
            with patch.dict('sys.modules',{'bert_score':SimpleNamespace(score=score)}):
                pilot.bertscore(run)
            metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['bertscore']['f1'],0.8)
            self.assertEqual(metrics['bertscore']['hash'],'test-hash')

    def test_saved_failed_draft_is_revised_and_reference_is_not_sent(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            run=Path(directory)
            out=run/'get_out'
            (out/'pipeline').mkdir(parents=True)
            pilot.write_json(out/'pipeline/checkpoint.json',{'narratives':[{'narrative':'evidence only'}]})
            pilot.write_json(out/'generation_attempts.json',{'latest_draft':'old '*775,'attempts':[]})
            (run/'summary_instruction.txt').write_text('Use only this narrative: ',encoding='utf-8')
            (out/'reference_summary.txt').write_text('REFERENCE_MARKER',encoding='utf-8')
            def response(content):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content),finish_reason='stop')],usage=None,model='mock')
            client=Mock()
            client.chat.completions.create.side_effect=[response(json.dumps({'paragraphs':['word '*140]*5})),response('{"overall_assessment":"mock review"}')]
            with patch('openai.AzureOpenAI',return_value=client), patch('src.path_utils.load_kairos_env'), \
                    patch.dict('os.environ',{'GPT_VERSION':'mock','GPT_ENDPOINT':'https://example.test','GPT_KEY':'mock','GPT_DEPLOYMENT':'mock'}):
                pilot.generate_and_score(run,'get_out')
            self.assertEqual(len((out/'generated_summary.txt').read_text().split()),700)
            messages=client.chat.completions.create.call_args_list[0].kwargs['messages']
            self.assertNotIn('REFERENCE_MARKER',json.dumps(messages))
            self.assertIn('Remove approximately 75 words',messages[-1]['content'])
            self.assertTrue(list(out.glob('generation_attempts_before_resume_*.json')))

    def test_skip_persists_and_each_failure_allows_next_movie(self):
        scratch = ROOT / '.tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pilot_queue_', dir=scratch) as directory:
            run = Path(directory).resolve()
            pilot.write_json(run / 'manifest.json', {'input_completeness': 'requires_confirmation'})
            pilot.write_json(run / 'status.json', {'state': 'prepared', 'movies': {}, 'updated_utc': pilot.now()})
            for slug, _, _ in pilot.MOVIES:
                (run / slug).mkdir()
            with patch.object(pilot.subprocess, 'Popen', side_effect=RuntimeError('simulated worker failure')) as launch, \
                    patch.object(pilot, 'bertscore'), contextlib.redirect_stdout(io.StringIO()):
                pilot.run_all(run, allow_unverified_inputs=True, skip=['yes_man'])
            status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
            manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(launch.call_count, 3)
            self.assertEqual(status['movies']['yes_man']['state'], 'skipped_by_user')
            self.assertTrue(all(status['movies'][s]['state'] == 'failed' for s in ['baby_driver','get_out','the_martian']))
            self.assertEqual(manifest['skipped_movies'], ['yes_man'])
            self.assertTrue(manifest['diagnostic_inputs_authorized'])
            self.assertEqual(status['state'], 'finished')
            # Resuming without flags must retain the user's exclusion and authorization.
            with patch.object(pilot.subprocess, 'Popen', side_effect=RuntimeError('simulated failure')) as resume, \
                    patch.object(pilot, 'bertscore'), contextlib.redirect_stdout(io.StringIO()):
                pilot.run_all(run)
            self.assertEqual(resume.call_count, 3)
            self.assertFalse(any('yes_man' in call.args[0] for call in resume.call_args_list))

    def test_unverified_inputs_do_not_launch_without_authorization(self):
        with tempfile.TemporaryDirectory(prefix='pilot_guard_', dir=ROOT / '.tmp') as directory:
            run = Path(directory).resolve()
            pilot.write_json(run / 'manifest.json', {'input_completeness': 'requires_confirmation'})
            with patch.object(pilot.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError, 'Input completeness'):
                    pilot.run_all(run)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
