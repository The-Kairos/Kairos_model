import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.rag_convo import embed_contexts


class EmbeddingBatchTest(unittest.TestCase):
    def test_gemini_caps_explicit_oversized_batch_and_preserves_order(self):
        client=Mock()
        client.models.embed_content.side_effect=lambda **kw: SimpleNamespace(
            embeddings=[SimpleNamespace(values=[int(value)]) for value in kw['contents']])
        result=embed_contexts([str(i) for i in range(225)],client=client,batch_size=999,provider='gemini')
        self.assertEqual([len(c.kwargs['contents']) for c in client.models.embed_content.call_args_list],[100,100,25])
        self.assertEqual(result,[[i] for i in range(225)])


if __name__=='__main__':
    unittest.main()
